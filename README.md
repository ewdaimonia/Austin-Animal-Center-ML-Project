# Austin Animal Center Survival ML Project

This is a small project which was created by following several tutorials in the Tensorflow documentation and applying them to a dataset from a local animal center. The idea was to write a console application that would create a model to predict animal survival rate based on several different features (breed, color, age, sex). Currently, the script will bring the csv data into a dataframe, then split that dataframe into train, validation, and test sets, convert those into datasets using keras, and then create a model and print out the predicted outcome and actual outcome for 32 entries in test dataset.
Found out that there is a second dataset of animal center intakes that closely matches this one. I am interested in seeing how our model changes with more feautures (length of stay, intake condition, intake type, location).  

Please note that I'm still relatively new to the data science world myself and certainly to machine learning, so if you see any additional issues with this script, please bring them up to me and I'd be very happy to try to remedy them. 

## Current issues

* ~~There is some issue with the data in the dog animal type entries which will not let us create a model for them~~
    - This issue has been fixed by returning 'Male' in a lambda function when it encounters the bad float data in the 'Sex upon Outcome' value. I would rather remove the value entirely, but I cannot find it in the xlsx file for the life of me.
* ~~Is sex upon outcome a good indicator? It may be the only factor which is used because the Austin Animal Center might offer spay and neuter services upon adoption, which would mean an adopted pet would almost always be spayed / neutered and a died / euthanized pet would almost never be. I am considering removing it from the feature column.~~ 
    - After some deliberation, I decided to remove the intact/neutered subtext from the column. It was too strong an indicator of the outcome type. I'm unsure why exactly almost all fixed animals are adopted and intact animals are euthanized, but I think it's more interesting to examine the other traits. I will reach out to the Austin Animal Care Center and ask if all adopted animals are required to be fixed.
	- Contacted staff at the Austin Animal Center, who informed me that all animals are required to be fixed upon adoption, thus it's good we dropped the prefix as intact would always indicate a deceased animal.

## Features to add
- [x] Make a utility script that combines the two datasets
- [ ] Use the new combined dataset to build a better model
- [X] User input so users can see a prediction of their animal's survival
- [ ] Split the createModel method into several smaller functions (must better follow the single responsibility principle)
- [ ] GUI
- [ ] Add some visualizations, maybe one that plots predicted survival over length of stay

## Tutorials followed:
* https://www.tensorflow.org/tutorials/structured_data/feature_columns

* https://www.tensorflow.org/tutorials/load_data/csv

* https://www.tensorflow.org/tutorials/load_data/pandas_dataframe

## Data source:
https://catalog.data.gov/dataset/austin-animal-center-intakes
https://catalog.data.gov/dataset/austin-animal-center-outcomes-version-1-demo
These datasets are updated frequently, datasets last pulled 8/24/2020
