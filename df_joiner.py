import pandas as pd
import numpy as np
import pprint as pp

# This is a small utility script to join the outcomes and intakes datasets into one larger dataset with more information to build our model on

outcomesdf = pd.read_csv('Austin_Animal_Center_Outcomes_new.csv')
# outcomesdf = outcomesdf.sort_values(
#     ['Animal ID', 'DateTime'], ascending=(False, True))
intakesdf = pd.read_csv('Austin_Animal_Center_Intakes.csv')
# intakesdf = intakesdf.sort_values(['Animal ID', 'DateTime'], ascending=(False, True))

outcomesdf = outcomesdf.rename(columns={"DateTime": "Datetime_of_Outcome",
                                        "MonthYear": "MonthYear_of_Outcome"})
intakesdf = intakesdf.rename(columns={"DateTime": "Datetime_of_Intake",
                                      "MonthYear": "MonthYear_of_Intake"})
outcomesdf["Occ_Number"] = outcomesdf.groupby("Animal ID").cumcount()+1
intakesdf["Occ_Number"] = intakesdf.groupby("Animal ID").cumcount()+1

intakesdf = intakesdf.drop(
    columns=['Name', 'Breed', 'Color', 'Animal Type'])

fulldf = pd.merge(intakesdf, outcomesdf, on=['Animal ID', 'Occ_Number'],
                  how='inner', validate="1:1")
fulldf = fulldf.drop(columns=['Occ_Number'])

pp.pprint(fulldf.head())
pp.pprint(fulldf.columns)
print(intakesdf.describe())
print(outcomesdf.describe())
print(fulldf.describe())

fulldf['Datetime_of_Outcome'] = pd.to_datetime(fulldf['Datetime_of_Outcome'])
fulldf['Datetime_of_Intake'] = pd.to_datetime(fulldf['Datetime_of_Intake'])
fulldf["Length_of_Stay"] = fulldf['Datetime_of_Outcome'] - \
    fulldf['Datetime_of_Intake']

fulldf.to_csv('Austin_Animal_Center_Full.csv', index=False)