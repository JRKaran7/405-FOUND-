import json
import tensorflow as tf 
import pandas as pd 
import ntlk 
import numpy as np
from tensorflow.keras. preprocessing. sequence import pad_sequences 
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense,GlobalMaxPooling1D,Flatten 
from tensorflow.keras. preprocessing. text import Tokenizer 
import pickle
import os



with open ("intents.json","r") as file:
    data = file.read()
intents = json.loads(data)


listed_patterns = []
listed_tags = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        listed_patterns.append(pattern)
        listed_tags.append(intent['tag'])
labels = pd.get_dummies(listed_tags).values            

        

vocab_size = 5000
embedding_dim = 64
max_length = 100
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"



tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer . fit_on_texts(listed_patterns)
word_index = tokenizer.word_index


sequences = tokenizer.texts_to_sequences(listed_patterns)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=4),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(vocab_size, activation='softmax')
])



model.compile (loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


num_epochs = 50
history = model.fit( labels , padded_sequences , epochs = num_epochs , verbose=2 , batch_size = 32, validation_split = 0.1 )


model.save( "F&N_intent_model.h5" ) 


with open( "F&N_tokenizer.pickle" , "wb" ) as file: 

    tokenizer = pickle.load (file)
