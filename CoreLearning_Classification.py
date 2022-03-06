import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

#Input function for training
def input_fn(features, labels, training=True, batch_size=256):
    
    #Convert the inputs to Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features),labels))
    
    if training:
        #Shuffle and repeat if you are in training mode.
        dataset = dataset.shuffle(1000).repeat()           

    return dataset.batch(batch_size)

#Input function for prediction
def input_fn_pred(features, batch_size=256):
    return  tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)
    

CVS_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'VersiColor', 'Virginica']

# Download cvs files from googleapis
#
#train_path= tf.keras.utils.get_file("iris_training.csv",
#                                     "http://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
#test_path= tf.keras.utils.get_file("iris_test.csv",
#                                     "http://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

#read data from files using pandas
#
#      SepalLength  SepalWidth  PetalLength  PetalWidth  Species
# 0            6.4         2.8          5.6         2.2        2
# 1            5.0         2.3          3.3         1.0        1
# 2            4.9         2.5          4.5         1.7        2
# 3            4.9         3.1          1.5         0.1        0
# 4            5.7         3.8          1.7         0.3        0
#
train = pd.read_csv('iris_training.csv', names=CVS_COLUMN_NAMES, header=0)
test = pd.read_csv('iris_test.csv', names=CVS_COLUMN_NAMES, header=0)

# Separate the 'Species' column
train_y = train.pop('Species')
test_y = test.pop('Species')

# Feature columns describe how to use the imput
my_feature_columns = []
for key in train.keys():   #Loop over ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
    
# BUILIDING THE CLASSIFICATION MODEL
# We are going to use a Depp Neural Network with 2 hiden layers with 30 and 10 hidden nodes each

classifier = tf.estimator.DNNClassifier(
    feature_columns = my_feature_columns,
    hidden_units=[30,10], #number of nodes
    n_classes=3)  #there are 3 classes: ['Setosa', 'VersiColor', 'Virginica']


#Training the model, we use a lambda function to pass the input_fn
classifier.train(
    input_fn = lambda: input_fn(train, train_y, training=True),
    steps=5000)
 
#Evaluation
eval_result=classifier.evaluate(input_fn = lambda: input_fn(test, test_y, training=False))

#Prediction

features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'] 
predict = {}

print("Type features")
for feature in features:
    valid = True
    while valid:
        val = input(feature + ": ")
        if not val.isdigit(): valid=False

    predict[feature]=[float(val)]

predictions = classifier.predict(input_fn = lambda: input_fn_pred(predict))

for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print('Prediction is "{}" ({:.1f}%)'.format(SPECIES[class_id], 100*probability))
