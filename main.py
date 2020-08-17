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

# This is the method which creates a model and then calls the make predictions method so the users can see what the prediction for their pet will be
# This method almost certainly needs to be split into one method that sanitizes the data set, one that creates the model, and one that handles the predictions.


def createModel(df):
    # This method was created by following the following tf tutorials:
    # https://www.tensorflow.org/tutorials/structured_data/feature_columns + https://www.tensorflow.org/tutorials/load_data/csv

    # Set up a dictionary of categorical lists to help with categorical features later
    CATEGORIES = {
        # 'Sex_upon_Outcome': ['Intact Male', 'Intact Female', 'Spayed Female', 'Neutered Male', 'Unknown'],
        'Sex_upon_Outcome': ['Male', 'Female'],
        'Breed': [*df['Breed'].unique()],
        'Color': [*df['Color'].unique()]
    }

    # Code to simplify age to a year value or a decimal value for under a year
    df['Age upon Outcome'] = df['Age upon Outcome'].apply(
        lambda age: float(age[0]) if 'year' in str(age) else(float(age[0])/12 if 'month' in str(age) else(float(age[0])/52.1429 if 'week' in str(age) else float(0))))

    # Remove entries with outcomes that don't make sense for our model
    # Also remove incomplete data such as NULL or Unknown entries for sex and age
    df = df[(df['Outcome Type'] != 'Transfer') & (
        df['Outcome Type'] != 'Return to Owner') & (df['Age upon Outcome'] != 'NULL') & (df['Sex upon Outcome'] != 'Unknown') & (df['Sex upon Outcome'] != 'NULL')]

    # Simplify outcomes(target column) to 1 for adopted / lived and 0 for died / euthanized
    df['Outcome Type'] = df['Outcome Type'].map(lambda out: 1 if out in [
        'Rto-Adopt', 'Return to Owner', 'Transfer', 'Adoption'] else 0)

    # Remove neutered / spayed / intact part of gender as it was too great of an indicator of survival.
    # I highly dislike the if statement in this lambda, but I could not find the source of the error with the dog animal type.
    # The script kept breaking on a float value which I cannot find in the actual data
    df['Sex upon Outcome'] = df['Sex upon Outcome'].apply(
        lambda sex: sex.split()[1] if isinstance(sex, str) else 'Male')

    # Drop unecessary columns
    df = df.drop(columns=['Animal ID', 'Name', 'DateTime',
                          'Date of Birth', 'MonthYear', 'Outcome Subtype', 'Animal Type'])

    # These three describe statements are useful because they allow us to see the averages / frequencies / means for the all the adopted vs died / euthanized entries
    # print(df[df['Outcome Type'] == 1].describe(include='all'))
    # print(df[df['Outcome Type'] == 0].describe(include='all'))
    # print(df['Outcome Type'].describe())
    # input()

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
        # Add our numerical and categorical layers
        feature_layer,
        # Been playing around with the units and number of layers to try to build a better model
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        # layers.Dense(128, activation='relu'),
        # I don't think dropout will work since we are mostly using categorical data here, setting some values to 0 would only work for the age column
        # layers.Dropout(.1),
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
    # Save our model for use elsewhere
    # model.save('aacmlp')

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

    makePrediction(model, df)


def makePrediction(model, df):
    #imported = tf.saved_model.load("aacmlp")
    userPet = []
    userPet.append(input(
        "Please enter the age of your animal in years, or decimal values for under a year: "))
    userPet.append(input(
        f"{', '.join(map(str, df['Breed'].unique()))}, Please enter the breed of your animal which most closely matches one of the options listed above: "))
    userPet.append(input(
        f"{df['Color'].unique()} Please enter the color of your animal which most closely matches one of the options listed above: "))
    userPet.append(input(
        "Please enter the gender of your animal [Male or Female]: "))
    singleDf = pd.DataFrame([{'Age_upon_Outcome': float(userPet[0]), 'Breed': str(userPet[1]),
                              'Color': str(userPet[2]), 'Sex_upon_Outcome': str(userPet[3]), 'target': float(0)}])
    singleDs = df_to_dataset(singleDf, batch_size=1)
    singlePrediction = model.predict(singleDs)
    singlePrediction = tf.sigmoid(singlePrediction[0]).numpy()
    print("Predicted survival: {:.2%}".format(singlePrediction[0]))

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
    userChoice = input(
        "Please pick one of the animal types to build a model for: [Cat, Dog, Bird, Other] ")
    if(userChoice in originaldf['Animal Type'].unique()):
        createModel(originaldf[(originaldf['Animal Type'] == userChoice)])
    else:
        print("Your entry did not match any of the choices, goodbye")
    # createVisualizations(catdf)


if __name__ == "__main__":
    main()
