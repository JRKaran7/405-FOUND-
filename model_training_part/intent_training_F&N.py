import json
import tensorflow as tf
import pandas as pd
from tensorflow.keras. preprocessing. sequence import pad_sequences # type: ignore
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense,GlobalMaxPooling1D,Flatten # type: ignore
from tensorflow.keras. preprocessing. text import Tokenizer # type: ignore
import pickle

with open ("model_training_part/intents.json","r") as file:
    data = file.read()
intents = json.loads(data)

listed_patterns = []
listed_tags = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        listed_patterns.append(pattern)
        listed_tags.append(intent['tag'])
labels = pd.get_dummies(listed_tags).values            

embedding_dim = 64
trunc_type='post'
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=3, oov_token=oov_tok)
tokenizer . fit_on_texts(listed_patterns)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(listed_patterns)
padded_sequences = pad_sequences(sequences, maxlen=40, padding='post', truncating=trunc_type)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(3, embedding_dim, input_length=40),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(64, 5, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=4),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile (loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

num_epochs = 50
history = model.fit(padded_sequences, labels, epochs = num_epochs , verbose=2 , batch_size = 32, validation_split = 0.1 )

model.save("generated_models/F&N_intent_model.h5")

with open("generated_models/F&N_tokenizer.pickle", "wb") as file:
    pickle.dump(tokenizer,file)
