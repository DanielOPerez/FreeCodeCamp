import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

# This function prepares de data for training. Creates a TensorFlow Dataset,
# By default reorganices the data in batchs of 32 elements, after shuffling
# Then creates several epochs
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function


# Load dataset using pandas
# This is a list of passangers of the Titanic, which survirved or not, with information
# about their age, class, sex, etc
#
#     survived     sex   age  n_siblings_spouses  ...   class     deck  embark_town alone
# 0           0    male  35.0                   0  ...   Third  unknown  Southampton     y
# 1           0    male  54.0                   0  ...   First        E  Southampton     y
# 2           1  female  58.0                   0  ...   First        C  Southampton     y
# 3           1  female  55.0                   0  ...  Second  unknown  Southampton     y
# 4           1    male  34.0                   0  ...  Second        D  Southampton     y
#

dftrain = pd.read_csv('http://storage.googleapis.com/tf-datasets/titanic/train.csv') #training dataset
dfeval = pd.read_csv('http://storage.googleapis.com/tf-datasets/titanic/eval.csv') #testing data

# Cut the information about their survival statu dftrain and dfeval
# and keep it in y_train and y_eval using .pop() from pandas

y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

# Separate between categorical data an numeric data. 

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

#It si necessary to code the categories using  numerical labels 
feature_columns=[]
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique() #gets a list of all unique values for a given column
    # fill feature_columns with the vocabulary corresponding to the list of feature_name.
    #
    # tf.feature_column.categorical_column_with_vocabulary_list(
    #  key, vocabulary_list, dtype=None, default_value=-1, num_oov_buckets=0)
    #
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))
    #now feature_colums replaces categorical columns, but with numbers instead of keys.

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))
    
#now feature_colums replaces categorical columns, but with numbers instead of keys

#CREATE THE MODEL. 


#Prepare the data for creating the model
train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False) #no need to suffle or epochs

#Training
linear_est = tf.estimator.LinearClassifier(feature_columns = feature_columns)
linear_est.train(train_input_fn) # train

#Evaluating
result = linear_est.evaluate(eval_input_fn)
