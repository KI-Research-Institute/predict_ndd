#############################################
# wrapper file for deploying ASD prediction
#############################################
import pandas as pd
import numpy as np
import os
import pickle
import timeit
import json

import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)

#from bl_deploy_utils import prepare_demo_data
datadir = "./data/"
model_dir = './data/models/'
model_list = {18:'autism_xgb_compact_548d_all',24:'autism_xgb_compact_730d_all',36:'autism_xgb_compact_1095d_all'}
date_format = '%Y-%m-%d'

def load_models(model_dir):
    models = dict()
    for age,model_name in model_list.items():
        model_fn = os.path.join(model_dir,f'{model_name}.pkl')
        with open(model_fn, 'rb') as fp:
            models[model_name] = pickle.load(fp)
    return models

def load_scale(datadir,scale_name='all_normals'):
    fn = os.path.join(datadir,f'scale_data_{scale_name}.csv')
    th_df = pd.read_csv(fn)
    return th_df

def calc_colors(col, age, th):
    colors = ['yellow', 'orange', 'red']
    out_col = col.copy()
    out_col[col == True] = 0
    out_col[col == False] = 1
    out_col_frac = out_col.copy()
    min_age = th[colors].min()
    out_col_frac[(col == False) & (age <= min_age)] = age / min_age

    for code, color in enumerate(colors):
        rows = (col == False) & (age > th[color])
        out_col[rows] = code + 2

    max_age = th['max_age']
    threshs = th[colors].values.tolist() + [max_age]
    # handle ages that are larger than the maximal age
    age_trim = age.copy()
    age_trim[col.notna() & (age>max_age)] = max_age

    for code, color in enumerate(colors):
        rows = (col == False) & (age_trim > threshs[code]) & (age_trim <= threshs[code + 1])
        out_col_frac[rows] = code + 1 + (age_trim - threshs[code]) / (threshs[code + 1] - threshs[code])

    return out_col, out_col_frac

def color_by_threshold(th_df, dev_df):

    type_to_name = {1: 'social', 2: 'language', 3: 'gross_motor', 4: 'fine_motor'}
    dev_df['AgeInMonths'] = dev_df['Age'] / (365 / 12)
    th = th_df.set_index('english')

    for task, task_old in zip(th_df['english_new'], th_df['english']):
        c, c_frac = calc_colors(dev_df[task], dev_df['AgeInMonths'], th.loc[task_old])
        dev_df[f'{task}_frac'] = c_frac

    return dev_df

def prepare_features(visit_data,demography_data,scale_data,convert_to_array=True):

    # calculate DSS features
    dev_df = color_by_threshold(scale_data, visit_data)
    age_steps = ['1_3','3_6','6_9','9_12','12_18','18_24','24_36']
    age_periods = [(0,6),(6,12),(12,18),(18, 24),(24,36)]
    domain_list = {'SO':'social','LC':'language','GM':'gross_motor','FM':'fine_motor'}
    name_to_type = {'social':1,'language':2,'gross_motor':3, 'fine_motor':4}
    max_age = 0

    features = pd.Series(dtype='float64')
    for age_period in age_periods:
        for type_name in ['gross_motor','fine_motor','language','social']: # note the order here matters
            feat_name = str((age_period, type_name))
            type_id = name_to_type[type_name]
            tasks = scale_data[(scale_data.type == type_id)]['english_new']
            tasks = [t + '_frac' for t in tasks]
            features[feat_name] = dev_df[dev_df['Age'].between(age_period[0]*(365/12),age_period[1]*(365/12))][tasks].stack().mean()

    for step in age_steps:
        for type_name in ['LC','SO','FM','GM']: # note the order here matters
            feat_name = f'{step}_{type_name}'
            type_id = name_to_type[domain_list[type_name]]
            tasks = scale_data[(scale_data.steps==step) & (scale_data.type==type_id)]['english_new']
            tasks = [t + '_frac' for t in tasks]
            features[feat_name] = dev_df[tasks].max(axis=0).mean() #mean over the tasks of max per task
            if ~np.isnan(features[feat_name]):
                max_age = max(max_age,int(step.split('_')[1]))


    # combine demographic features
    features['GenderDesc_Male'] = float(demography_data['GenderDesc']=='Male')
    features['Mother_age_G40'] = float(demography_data['Mother_age']>40)
    features['Mother_age_L20'] = float(demography_data['Mother_age']<20)
    features['Mother_age_Missing'] = float(demography_data['Mother_age'].isna())
    features['EducationLevelDesc_Elementary'] = float(demography_data['EducationLevelDesc']=='Elementary')
    features['EducationLevelDesc_High School'] = float(demography_data['EducationLevelDesc'] == 'High School')
    features['EducationLevelDesc_Missing'] = float(demography_data['EducationLevelDesc'] == 'Missing')
    features['EducationLevelDesc_Tertiary'] = float(demography_data['EducationLevelDesc'] == 'Tertiary')

    # drop redundant features
    features = features.drop(labels=['3_6_SO','6_9_SO','12_18_FM',"((12, 18), 'fine_motor')"])
    if max_age < 36:
        features = features.drop(labels=['24_36_LC','24_36_SO','24_36_GM','24_36_FM',"((24, 36), 'gross_motor')",
                                         "((24, 36), 'fine_motor')","((24, 36), 'language')","((24, 36), 'social')"])
    if max_age < 24:
        features = features.drop(labels=['18_24_LC','18_24_SO','18_24_GM','18_24_FM',"((18, 24), 'gross_motor')",
                                         "((18, 24), 'fine_motor')","((18, 24), 'language')","((18, 24), 'social')"])
    if max_age < 18:
        features = features.drop(labels=['12_18_LC','12_18_SO','12_18_GM',"((12, 18), 'gross_motor')",
                                         "((12, 18), 'language')","((12, 18), 'social')"])

    if convert_to_array:
        features = np.array(features.values)
        features = features.reshape(1,features.shape[0])

    # determine model type (compact or snapshot) and age
    if max_age in model_list.keys():
        model_name = model_list[max_age]
    else:
        model_name = None
        features = None

    #TODO: add sanity checks for the features and return different error codes/messages
    #TODO: check gestational age and return an error for preterms <34w
    #TODO: consider to use snapshot models

    return features, model_name

def predict(features, model):
    y_proba = model.predict_proba(features)

    # TODO: translate score to a summary message (based on threshold on the calibrated scores)
    return y_proba[:,1]

def read_json(fn,scale_data):

    with open(fn,'r') as f:
        json_data = json.load(f)

    # convert data
    demographic_variables = ['GenderDesc', 'Mother_age', 'EducationLevelDesc', 'PregnancyWeek']
    visit_variables = ['PersonId','Age'] + list(scale_data.english_new.values)

    demography_data = pd.DataFrame(columns = demographic_variables)
    visit_data = pd.DataFrame(columns = visit_variables)

    dob = pd.to_datetime(json_data['patient']['dob'],format=date_format)
    mother_dob = pd.to_datetime(json_data['patient']['motherDob'],format=date_format)
    mother_age = (dob - mother_dob).days/365
    demography_data = demography_data.append({'GenderDesc':json_data['patient']['sex'],
                                              'Mother_age':mother_age,
                                              'EducationLevelDesc':json_data['patient']['motherEducation'],
                                              'PregnancyWeek':json_data['patient']['pregnancyWeek']},ignore_index=True)

    empty_visit = dict.fromkeys(visit_variables)
    for v in json_data['visits']:
        visit = empty_visit.copy()
        visit['PersonId'] = json_data['patient']['id']
        visit['Age'] = (pd.to_datetime(v['date'],format=date_format) - dob).days
        for m in v['milestones']:
            if m['result'] == 'pass':
                visit[m['name']] = True
            elif m['result'] == 'fail':
                visit[m['name']] = False
        visit_data = visit_data.append(visit,ignore_index=True)

    return demography_data,visit_data

if __name__ == "__main__":
    # usage example

    # load models and scale data
    models = load_models(model_dir)
    scale_data = load_scale(datadir)

    # run tests
    tests_dir = os.path.join(datadir,'tests/')
    test_ids = os.listdir(tests_dir)
    test_df = pd.DataFrame()
    for fn in test_ids:
        dn = os.path.join(tests_dir,fn)
        print(dn)
        demography_data,visit_data = read_json(dn,scale_data)

        start = timeit.default_timer()
        features, model_name = prepare_features(visit_data, demography_data,scale_data)
        if (not features is None) & (model_name in models.keys()):
            score = predict(features, models[model_name])
            end = timeit.default_timer()
            print(model_name,score,f'{(end-start):.2f}s')
            test_df = test_df.append({'id':float(os.path.splitext(fn)[0]),'score':score[0]},ignore_index=True)
        else:
            print('Error: failed to prepare features')

    exit(0)
