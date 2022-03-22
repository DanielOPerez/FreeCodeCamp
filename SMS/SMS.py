import tensorflow as tf
import pandas as pd
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

train_file_path = "train-data.tsv"
test_file_path = "valid-data.tsv"

train_dataset=pd.read_csv(train_file_path, sep='\t',names=['type', 'sms'])
test_dataset=pd.read_csv(test_file_path, sep='\t',names=['type', 'sms'])

train_data = train_dataset['sms']
train_labels = train_dataset['type']

test_data = test_dataset['sms']
test_labels = test_dataset['type']

#======================================================================================================
#Tokenization of the text data
#======================================================================================================

num_words = 3000         #vocabulary size
oov_token = '<UNK>'
pad_type = 'pre'
trunc_type = 'post'

#create tokenizer and apply to train_data
tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
tokenizer.fit_on_texts(train_data)
# Get our training data word index
word_index = tokenizer.word_index
# Encode training data sentences into sequences
train_sequences = tokenizer.texts_to_sequences(train_data)
# Get max training sequence length
maxlen = max([len(x) for x in train_sequences])
# Pad the training sequences
train_padded = pad_sequences(train_sequences, padding=pad_type, truncating=trunc_type, maxlen=maxlen)

#Tokenize test data
test_sequences = tokenizer.texts_to_sequences(test_data)
test_padded = pad_sequences(test_sequences, padding=pad_type, truncating=trunc_type, maxlen=maxlen)

#Encode labels
train_labels=pd.factorize(train_labels)[0]
test_labels=pd.factorize(test_labels)[0]

#create the model

model = tf.keras.Sequential([
     tf.keras.layers.Embedding(num_words, 32),
     tf.keras.layers.LSTM(32),
     tf.keras.layers.Dense(1, activation='sigmoid')
])

#Training the model
model.compile(loss="binary_crossentropy",
              optimizer="rmsprop",
              metrics=["acc"])

history = model.fit(train_padded,                    
                    train_labels,
                    epochs=10,
                    validation_split=0.2)

#model.save('SMS_model.h5')
#model = load_model('SMS_model.h5')


#evaluate the model
result = model.evaluate(test_padded,test_labels)
print(result)

# function to predict messages based on model
# (should return list containing prediction and label, ex. [0.008318834938108921, 'ham'])
def predict_message(pred_text):

    encoded_text=tokenizer.texts_to_sequences([pred_text])
    encoded_padded = pad_sequences(encoded_text, padding=pad_type, truncating=trunc_type, maxlen=maxlen)
    prediction = model.predict(encoded_padded)

    ham_or_spam='spam'
    if prediction <= 0.5:
        ham_or_spam='ham'
        
    return [float(prediction), ham_or_spam]

pred_text = "You won! here is your cash for hoy holidays?"

prediction = predict_message(pred_text)
print(prediction)


# Run this cell to test your function and model. Do not modify contents.
# def test_predictions():
#     test_messages = ["how are you doing today",
#                      "sale today! to stop texts call 98912460324",
#                      "i dont want to go. can we try it a different day? available sat",
#                      "our new mobile video service is live. just install on your phone to start watching.",
#                      "you have won £1000 cash! call to claim your prize.",
#                      "i'll bring it tomorrow. don't forget the milk.",
#                      "wow, is your arm alright. that happened to me one time too"
#                      ]
    
#     test_answers = ["ham", "spam", "ham", "spam", "spam", "ham", "ham"]
#     passed = True
    
#     for msg, ans in zip(test_messages, test_answers):
#         prediction = predict_message(msg)
#     if prediction[1] != ans:
#         passed = False
        
#     if passed:
#         print("You passed the challenge. Great job!")
#     else:
#         print("You haven't passed yet. Keep trying.")
        
# test_predictions()
                
