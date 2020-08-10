import pandas as pd
import numpy as np
import tensorflow as tf
import re
from tensorflow import feature_column
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

originaldf = pd.read_csv('Copy of Austin_Animal_Center_Outcomes.csv')

# Split the dataset into different animal types
catdf = originaldf[(originaldf['Animal Type'] == 'Cat')]
dogdf = originaldf[(originaldf['Animal Type'] == 'Dog')]
birddf = originaldf[(originaldf['Animal Type'] == 'Bird')]
otherdf = originaldf[(originaldf['Animal Type'] == 'Other')]

# A utility method to create a tf.data dataset from a Pandas Dataframe
# This method is entirely taken from the tf documentation


def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds

# A utility method to show batchs for troubleshooting
# This method is entirely taken from the tf documentation


def show_batch(dataset):
    for batch, label in dataset.take(1):
        for key, value in batch.items():
            print("{:20s}: {}".format(key, value.numpy()))


def createModelMixedCatMethod(df):
    # This method was created by following the following tf tutorials:
    # https://www.tensorflow.org/tutorials/structured_data/feature_columns + https://www.tensorflow.org/tutorials/load_data/csv

    # Set up a dictionary of categorical lists to help with categorical features later
    CATEGORIES = {
        'Sex_upon_Outcome': ['Intact Male', 'Intact Female', 'Spayed Female', 'Neutered Male', 'Unknown'],
        'Breed': [*df['Breed'].unique()],
        'Color': [*df['Color'].unique()]
    }

    # Code to simplify age to a year value or 0 for under a year
    # df['Age upon Outcome'] = df['Age upon Outcome'].apply(
    #     lambda age: 0 if 'year' not in str(age) else(int(age[0]) if age else 0))

    # Code to simplify age to a year value or a decimal value for under a year
    df['Age upon Outcome'] = df['Age upon Outcome'].apply(
        lambda age: float(age[0]) if 'year' in str(age) else(float(age[0])/12 if 'month' in str(age) else(float(age[0])/52.1429 if 'week' in str(age) else float(0))))

    # Remove entries with outcomes that don't make sense for our model
    df = df[(df['Outcome Type'] != 'Transfer') & (
        df['Outcome Type'] != 'Return to Owner')]
    # Simplify outcomes(target column) to 1 for adopted / lived and 0 for died / euthanized
    df['Outcome Type'] = df['Outcome Type'].map(lambda out: 1 if out in [
        'Rto-Adopt', 'Return to Owner', 'Transfer', 'Adoption'] else 0)

    # Drop unecessary columns
    df = df.drop(columns=['Animal ID', 'Name', 'DateTime',
                          'Date of Birth', 'MonthYear', 'Outcome Subtype', 'Animal Type'])
    # Replace outcome type column with target column
    df['target'] = df.pop('Outcome Type')
    # Replace whitespace with underscores in column names
    df.columns = [c.replace(' ', '_') for c in df.columns]
    # Split the dataframe into test, train, and validation sets
    train, test = train_test_split(df, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)
    # Show the size of each split df
    # print(len(train), 'train examples')
    # print(len(val), 'validation examples')
    # print(len(test), 'test examples')

    # Declare the feature columns list, this will hold all our columns that will dictate the creation of our model
    feature_columns = []

    # Add numeric cols
    feature_columns.append(
        feature_column.numeric_column('Age_upon_Outcome'))

    # Declare a temporary categorical columns list which will be iterated through and added to our feature columns
    categorical_columns = []
    for feature, vocab in CATEGORIES.items():
        cat_col = feature_column.categorical_column_with_vocabulary_list(
            key=feature, vocabulary_list=vocab)
        categorical_columns.append(feature_column.indicator_column(cat_col))

    for col in categorical_columns:
        feature_columns.append(col)

    # Create the feature layer which will include our feature columns and will help us build our model
    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

    # Set batch size and use the tf df to dataset utility function to create datasets we can train our model on
    batch_size = 32
    train_ds = df_to_dataset(train, batch_size=batch_size)
    val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

    # Use keras sequential to build our model
    model = tf.keras.Sequential([
        feature_layer,
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dropout(.1),
        layers.Dense(1)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_ds,
              validation_data=val_ds,
              epochs=10)

    loss, accuracy = model.evaluate(test_ds)
    print("Accuracy", accuracy)

    test_loss, test_accuracy = model.evaluate(test_ds)

    print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))

    predictions = model.predict(test_ds)

    # Show some results
    # We are using count here to iterate through the values in the test ds so we can show which features were used as input for a certain prediction
    count = 0
    for prediction, survived in zip(predictions[:32], list(test_ds)[0][1][:32]):
        # Uncomment this to only show entries with an outcome of 0
        # if(bool(survived)):
        #     count += 1
        #     continue
        prediction = tf.sigmoid(prediction).numpy()
        print("Gender:" + str(list(test_ds.as_numpy_iterator())
                              [0][0]['Sex_upon_Outcome'][count]))
        print("Age:" + str(list(test_ds.as_numpy_iterator())
                           [0][0]['Age_upon_Outcome'][count]))
        print("Breed:" + str(list(test_ds.as_numpy_iterator())
                             [0][0]['Breed'][count]))
        print("Color:" + str(list(test_ds.as_numpy_iterator())
                             [0][0]['Color'][count]))
        print("Predicted survival: {:.2%}".format(prediction[0]),
              " | Actual outcome: ",
              ("SURVIVED" if bool(survived) else "DIED"))
        print()
        count += 1

# This was an older method I used before implimenting categorical features, I should probably remove it, but I think it's a good example of how you can simplify data to numeric representations


def createModelCatNumsMethod(df):
    # This method was created by following the following tf tutorials:
    # https://www.tensorflow.org/tutorials/load_data/pandas_dataframe

    # Create a categorical representation of each gender option
    df['Sex upon Outcome'] = pd.Categorical(df['Sex upon Outcome'])
    df['Sex upon Outcome'] = df['Sex upon Outcome'].cat.codes
    # Change each value in the age column to either 0 if the age is < 1 year and the number of years if the age is > 1
    df['Age upon Outcome'] = df['Age upon Outcome'].apply(
        lambda age: 0 if 'year' not in str(age) else(int(age[0]) if age else 0))
    # Create a categorical representation of each breed option
    df['Breed'] = pd.Categorical(df['Breed'])
    df['Breed'] = df['Breed'].cat.codes
    # Create a categorical representation of each color option
    df['Color'] = pd.Categorical(df['Color'])
    df['Color'] = df['Color'].cat.codes
    # Remove transferred outcome rows and returned to owner as they overly complicate the data and it isn't clear how much the owner's attachment to the pet weighed into picking it up and transferred does not indicate the final outcome
    df = df[(df['Outcome Type'] != 'Transfer') & (
        df['Outcome Type'] != 'Return to Owner')]
    # Simplify each outcome as 1 or 0, where 1 = adopted and 0 = died or euthanized
    df['Outcome Type'] = df['Outcome Type'].map(lambda out: 1 if out in [
        'Rto-Adopt', 'Return to Owner', 'Transfer', 'Adoption'] else 0)

    # Separate our target column from the rest of the features in the df
    target = df.pop('Outcome Type')
    # Drop all columns that have no bearing on our model
    df = df.drop(columns=['Animal ID', 'Name', 'DateTime',
                          'Date of Birth', 'MonthYear', 'Outcome Subtype', 'Animal Type'])
    # Display the prepared df to create our ML model
    print(df.head())
    print(df.dtypes)

    # ML magic, idk, look at the TF docs
    dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))
    for feat, targ in dataset.take(1000):
        print(f'Features: {feat}, Target: {targ}')
    train_dataset = dataset.shuffle(len(df)).batch(1)

    def get_compiled_model():
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.BinaryCrossentropy(
                          from_logits=True),
                      metrics=['accuracy'])
        return model

    model = get_compiled_model()
    model.fit(train_dataset, epochs=3)

# This is a method that I'd like to flesh out a bit and add a few more visualiztions for examples of what can be done with data


def createVisualizations(df):
    df = df[(df['Outcome Type'] != 'Transfer') & (
        df['Outcome Type'] != 'Return to Owner')]
    df.drop
    fig = plt.figure()
    ax = plt.axes()
    ax.set(xlim=(0, 6), ylim=(0, 21000),
           xlabel='Outcome Type', ylabel='Frequency',
           title='First Visualization')

    plt.hist(df['Outcome Type'])


def main():
    # createModelCatNumsMethod(catdf)
    createModelMixedCatMethod(catdf)
    # createVisualizations(catdf)


if __name__ == "__main__":
    main()
