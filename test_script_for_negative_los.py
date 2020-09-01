import pandas as pd
import datetime

# Utility script to create a new csv with all negative length of stays

df = pd.read_csv('Austin_Animal_Center_Full.csv')
df['Length_of_Stay'] = pd.to_timedelta(df['Length_of_Stay'])
df = df[df['Length_of_Stay'] < datetime.timedelta(0)]
df.to_csv("baddates.csv", index=False)
