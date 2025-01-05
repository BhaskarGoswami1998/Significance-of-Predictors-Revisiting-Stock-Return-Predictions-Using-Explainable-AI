# Significance-of-Predictors-Revisiting-Stock-Return-Predictions-Using-Explainable-AI
This repository contains the code, datasets, and analysis for the paper "Significance of Predictors: Revisiting Stock Return Predictions Using Explainable AI". 

## Requirements
- `numpy` >= 1.19
- `pandas` >= 1.0.1
- `scikit-learn` >= 0.22.0
- `matplotlib` >= 3.1.0
- `tensorflow` == 2.5.0
- `keras` == 2.3.1

## Data
The initial raw daily data can be downloaded from [Here](#).  
*Caution:* This data set is very large (about 330MB) and consists of 3,147,312 rows and 14 columns. The data includes 1035 unique TIC IDs for firms.

Running the `[Data_clean.py]` script will produce a clean pickle file, namely `daily_clear_ret.pickle`, containing a pandas data frame. The clean data shape is `(928,773, 27)`, with 25 features and `TIC` and `CUSIP` as identifying variables.

## The Model
A Keras implementation of the Dy-Gap model proposed in the paper is presented in [Dy_Gap_Model.ipynb](#).


Data Processing (**Preprocessing.ipynb**): Scripts for cleaning, preprocessing, structuring financial datasets, and handling of missing data. Code to align predictors with stock return data over various historical periods. Applied PCA on Macroeconomic variables and interaction terms of the first 3 PCs with the original predictors is created. The original predictors are also part of the final dataset.<br />
<br />
Predictive Models (**Predictive Performance.ipynb**): Scripts where the battery of ML models (Linear Regression, Extra Tree Regressor, Random Forest Regressor, XGBoost Regressor, Set of Neural Networks) are used to predict the variety of dependent variables. Their prediction metrics are reported in the form of Mean Absolute Error, Mean Squared Error, and R2 Score both in-sample (on the train set) and out-of-sample (on the test set).
