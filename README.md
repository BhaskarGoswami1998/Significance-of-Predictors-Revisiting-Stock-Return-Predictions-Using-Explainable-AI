# Significance-of-Predictors-Revisiting-Stock-Return-Predictions-Using-Explainable-AI
This repository contains the code, datasets, and analysis for the paper "Significance of Predictors: Revisiting Stock Return Predictions Using Explainable AI". 

Data Processing (**Preprocessing.ipynb**): Scripts for cleaning, preprocessing, structuring financial datasets, and handling of missing data. Code to align predictors with stock return data over various historical periods. Applied PCA on Macroeconomic variables and interaction terms of the first 3 PCs with the original predictors is created. The original predictors are also part of the final dataset.<br />
<br />
Predictive Models (**Predictive Performance.ipynb**): Scripts where the battery of ML models (Linear Regression, Extra Tree Regressor, Random Forest Regressor, XGBoost Regressor, Set of Neural Networks) are used to predict the variety of dependent variables. Their prediction metrics are reported in the form of Mean Absolute Error, Mean Squared Error, and R2 Score.
