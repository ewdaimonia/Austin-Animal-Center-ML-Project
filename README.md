# Austin Animal Center Survival ML Project

This is a small project which was created by following several tutorials in the Tensorflow documentation and applying them to a dataset from a local animal center. The idea was to write a console application that would create a model to predict animal survival rate based on several different features (breed, color, age, sex). Currently, the script will bring the csv data into a dataframe, then split that dataframe into train, validation, and test sets, convert those into datasets using keras, and then create a model and print out the predicted outcome and actual outcome for 32 entries in test dataset. 

Please note that I'm still relatively new to the data science world myself and certainly to machine learning, so if you see any additional issues with this script, please bring them up to me and I'd be very happy to try to remedy them. 

## Current issues

* There is some issue with the data in the dog animal type entries which will not let us create a model for them
* Is sex upon outcome a good indicator? It may be the only factor which is used because the Austin Animal Center might offer spay and neuter services upon adoption, which would mean an adopted pet would almost always be spayed / neutered and a died / euthanized pet would almost never be. I am considering removing it from the feature column. 

## Features to add

* User input so users can see a prediction of their animal's survival
* GUI
* Add some visualizations

## Tutorials followed:
* https://www.tensorflow.org/tutorials/structured_data/feature_columns

* https://www.tensorflow.org/tutorials/load_data/csv

* https://www.tensorflow.org/tutorials/load_data/pandas_dataframe

## Data source:
https://catalog.data.gov/dataset/austin-animal-center-intakes
