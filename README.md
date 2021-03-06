# Airbnb Recommendations

The code here is for training, validating, and deploying an Airbnb Country Recommendation System, 
using an XGB Classifier and SkLearn packages.

## What's the Airbnb Recommendations?
Our Airbnb Recommendations is a machine learning library for predicting the new countries that a new user
will book an airbnb in. There are a total of 10+ countries of interest, that need to be recommended given
the Airbnb dataset. 

## Set up

### System Requirements
1. Python 3.6 or higher
2. Python libraries: pandas, scikit-learn, matplotlib, numpy, tqdm, xgb

#### Conda Setup Instructions
If you are using conda, you can get started by cloning this repository and using the 
environment.yaml file as in the following:
```bash
conda env create -n airbnb_env -f env.yml
```
and then activate the environment using the following,
```bash
conda activate airbnb_env
```

#### Docker Setup Instructions
If you are using docker, there is a Dockerfile included. 
After cloning this repository, you can build the image with the following,
```bash
docker build -t airbnb_recommender .
```
and then run it locally using,
```bash
docker run -it -p 8501:8501 airbnb_recommender
```

### Download required files
Download the [dataset](https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings/data) and store it within the project dir as `airbnb-recruiting-new-user-bookings`.
Next create the dataset. Here we will create our engineered features and process
 the dataset.
```bash
cd data
python make_dataset.py
```
Doing this, you'll output a processed baseline csv file `airbnb-recruiting-new-user-bookings/train_users_2-processed.csv`

## Train and evaluate model
```bash
cd <PROJECT_DIR>
python train_model.py
```

## Running streamlit application
```bash
streamlit run src/navigation_app.py
```

Directory Structure
------------

The directory structure of the Airbnb Recommendations project looks like this: 

```
├── README.md                               <- The top-level README for developers using this project.
├── data                                    <- Scripts to download or generate data
│   ├── merge_baseline_sessions.py
│   ├── d_utils.py
│   └── make_dataset.py
├── sessions-data                           <- Scripts to preprocess and feature engineer sessions data
│   ├── create-sessions-casting+ratio-csv.py
│   ├── merge-sessions.py
│   ├── pearson-features.py
│   └── generate_session_distinct_counts_and_time_features.py
├── models                                  <- Scripts to select and evaluate model
│   ├── select_model.py
│   └── eval_model.py
├── airbnb-recruiting-new-user-bookings     <- Expected data file
│
└── train_model.py                          <- Main script
```
