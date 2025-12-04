import json
import tensorflow as tf
import pandas as pd

from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.layers import Embedding, Dropout, Conv1D, MaxPooling1D, LSTM, Dense # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau # type: ignore
import pickle
import os


os.makedirs("generated_models", exist_ok=True)

# it is opening the intents.json that its doing
try:
    with open("model_training_part/intents.json", "r") as file:
     intents = json.load(file)
except FileNotFoundError:
    exit(1)


# list is created for patterns and tags 
patterns = []
tags = []

# for loops 

for intent in intents['intents']:
    
    for pattern in intent['patterns']:
        
        patterns.append(pattern)
        tags.append(intent['tag'])



# labels 
labels_df = pd.get_dummies(tags)

labels = labels_df.values

num_classes = labels.shape[1]


label_list = list (labels_df.columns)


# label_list 
with open( "generated_models/F&N_labels.pickle", "wb" ) as f:
    pickle.dump(label_list, f)


oov_token = "<OOV>" 

max_words = 1000

max_length = 40

tokenizer = Tokenizer( num_words=max_words, oov_token=oov_token )

tokenizer.fit_on_texts(patterns)

word_index = tokenizer.word_index


# sequences & padded sequences

sequences = tokenizer.texts_to_sequences(patterns)
padded_sequences = pad_sequences(sequences, maxlen=max_length, 
                                 padding='post', truncating='post')



embedding_dim = 64

model = tf.keras.Sequential([
    Embedding(input_dim=max_words,    output_dim=embedding_dim,      input_length=max_length,  name="embedding"),     Dropout(0.2, name="dropout_1"),
    Conv1D(64, 5, padding='same',     activation='relu', name="conv1d"),           MaxPooling1D(pool_size=4, name="maxpool"),
    LSTM(64, name="lstm"),            Dense(64, activation='relu',    name="dense_1"),              Dropout(0.3, name="dropout_2"),  
    Dense( num_classes,  activation='softmax' , name="output" )
])


model.compile(
            loss='categorical_crossentropy',  
            optimizer='adam',
    metrics=['accuracy']
)

model.summary()


callbacks = [
    EarlyStopping( monitor='val_loss', 
                  patience = 20 , restore_best_weights = True, verbose=1 ),
    
    ModelCheckpoint( 'generated_models / F&N_intent_model_best.h5', 
                    monitor='val_accuracy',save_best_only=True, verbose=1 ),

 ReduceLROnPlateau( monitor='val_loss', 
                   factor=0.5, patience=8, min_lr=0.00001, verbose=1 )                    ]


history = model.fit( padded_sequences,        labels,            epochs=200,    
batch_size=8,           validation_split=0.2,               callbacks=callbacks,
 verbose=1 )




final_loss = history.history['loss'][-1]
final_acc = history.history['accuracy'][-1]

final_val_loss = history.history['val_loss'][-1]

final_val_acc = history.history['val_accuracy'][-1]



model.save("generated_models/F&N_intent_model.h5")


with open("generated_models/F&N_tokenizer.pickle", "wb") as f:
    pickle.dump(tokenizer, f)


config = {
    'max_words': max_words,          'max_length': max_length,
    'embedding_dim': embedding_dim,  'num_classes': num_classes,     'labels': label_list,
    'vocab_size': len(word_index),   'num_patterns': len(patterns) }

with open( "generated_models /F&N_config.json" , "w" ) as f:
    
    json.dump  (config, f, indent=2 )
