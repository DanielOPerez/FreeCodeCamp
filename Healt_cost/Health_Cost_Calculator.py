import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

import seaborn as sns

from tensorflow import keras
from tensorflow.keras import layers


# Load dataset using pandas
#        age     sex   bmi  children smoker     region  expenses
#  1333   50    male  31.0         3     no  northwest  10600.55
#  1334   18  female  31.9         0     no  northeast   2205.98
#  1335   18  female  36.9         0     no  southeast   1629.83
#  1336   21  female  25.8         0     no  southwest   2007.95
#  1337   61  female  29.1         0    yes  northwest  29141.36
#
dataset = pd.read_csv('insurance.csv')


CATEGORICAL_COLUMNS = ['sex', 'smoker', 'region']
INT_COLUMNS = ['age', 'children']
NUMERIC_COLUMNS = ['bmi']

# Preprocessing data using pandas. The strings columns must be
# coded as integers. To do this I'm using pandas factorize
for feature_name in CATEGORICAL_COLUMNS:
    dataset[feature_name]=pd.factorize(dataset[feature_name])[0]

#for feature_name in INT_COLUMNS:
#    dataset[feature_name]=pd.factorize(dataset[feature_name])[0]    

#We are going to use 80% of the data to train and 20% to eval
#the data is goint to be randomly selected from the dataset

#Randomly select 80% of dataset to train
train_dataset=dataset.sample(frac=0.8)
#Drop the remainen 20% to eval
test_dataset=dataset.drop(train_dataset.index)

#define features and labels

train_features = train_dataset.copy()
test_features = test_dataset.copy()
# Cut the information about their expenses
# and keep it in y_train and y_eval using .pop() from pandas
train_labels = train_dataset.pop('expenses')
test_labels = test_dataset.pop('expenses')


# Numerical columns
# Create the normalization layer.
# Calculate a mean and variance for each index on the last axis.
# Fit the state of the preprocessing layer to the data by calling Normalization.adapt:
# This layer has the mean an var of the train_features, when applied to a given input data
# the result is (input_data - mean(train_features))/var(train_features)
    
normalizer = layers.Normalization(axis=-1)
normalizer.adapt(train_dataset)

#Create the model

model = tf.keras.Sequential([
    normalizer,
    layers.Dense(16),
    layers.Dense(8),
    layers.Dense(1)
    ])

#Compile the model

model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.1),
              loss='mae',
              metrics=['mae', 'mse'])
model.build()
model.summary()
#Training the model

model.fit(train_dataset,
          train_labels,
          validation_split=0.5,
          epochs=50,
          verbose=1)

    
# RUN THIS CELL TO TEST YOUR MODEL. DO NOT MODIFY CONTENTS.
# Test model by checking how well the model generalizes using the test set.

loss, mae, mse = model.evaluate(test_dataset, test_labels)

print("Testing set Mean Abs Error: {:5.2f} expenses".format(mae))

if mae < 3500:
  print("You passed the challenge. Great job!")
else:
  print("The Mean Abs Error must be less than 3500. Keep trying.")

# Plot predictions.
# test_predictions = model.predict(test_dataset).flatten()

# a = plt.axes(aspect='equal')
# plt.scatter(test_labels, test_predictions)
# plt.xlabel('True values (expenses)')
# plt.ylabel('Predictions (expenses)')
# lims = [0, 50000]
# plt.xlim(lims)
# plt.ylim(lims)
# # _ = plt.plot(lims,lims)



