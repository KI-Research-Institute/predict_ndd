# predict_ndd
Prediction of neuro-developmental delays using routine developmental surveillance data

Supplementary to the paper: Early Prediction of Autistic Spectrum Disorder Using Developmental Surveillance Data
  https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2813801

# Installation
create and activate a conda environment ('predict_ndd') with the required packages:
```
conda env create -f predict_ndd_env.yml
conda activate predict_ndd
```

test the package by running:
```
python predict_ndd.py
```

# Overview
predict_ndd is a code package for predicting the likelihood of autistic spectrum disorder (ASD) using data from assessments of age-appropriate milestone attainment.
## Directories and files:

predict_ndd.py - main code file, runs the prediction on the example input files

data/scale_data_all_normals.csv - age norms for milestone attainment by the THIS developmental scale

data/models/*.pkl - prediction models

data/tests/*.json - example input files for testing

## Input format (JSON):

The input JSON consists of a "patient" section with demographic information and an array of "visits".
A visit consits of a date and an array of milestones and the result of their assessment. 
```
{
  "patient": {
    "id": "123",
    "dob": "ddmmyyyy",
    "sex": "male",
    "pregnancy_week": 38,
    "mother_dob": "ddmmyyyy",
    "mother_education": "academic"
  },
  "visits": [
    {
      "date": "ddmmyyyy",
      "milestones": [
        {
          "name": "SmilesSocialSmile",
          "result": "perform"
        },
        {
          "name": "RespondsToVoicesAndSounds",
          "result": "perform"
        },
        {
          "name": "HoldsOrReachesHandsToObjecy",
          "result": "not_perform"
        }
      ]
    },
    {
      "date": "ddmmyyyy",
      "milestones": [
        {
          "name": "SitsByItself",
          "result": "perform"
        },
        {
          "name": "MakesSoundsInConverstation",
          "result": "perform"
        },
        {
          "name": "WalksWithHelp",
          "result": "not_perform"
        }
      ]
    },
...
  ]
}
```
The names of the milestones for each age step can be obtained from ./data/scale_data_all_normals.csv

## Usage flow:
1. Initialization by loading the prediction models and the scale data
```
models = load_models(model_dir)
scale_data = load_scale(datadir)
```
2. Reading the input file and preparing the prediction features
```
demography_data,visit_data = read_json(input_directory,scale_data)
features, model_name = prepare_features(visit_data, demography_data,scale_data)
```
3. Running the prediciton
```
score = predict(features, models[model_name])
```
The returned score (between 0 and 1) is the likelihood of ASD. 

# Citation
Amit, G., Bilu Y., et al. Early Prediction of Autistic Spectrum Disorder Using Developmental Surveillance Data. JAMA Netw Open 7, e2351052 (2024).



