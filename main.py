import pandas as pd
import io
import json
import numpy as np
import os
import re
from numpy import asarray
import random
from filter_review import *
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import one_hot, Tokenizer, tokenizer_from_json
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model, Model
from keras.layers import Flatten, Input, Bidirectional, GlobalMaxPooling1D, Embedding, Conv1D, LSTM, MaxPooling1D, SpatialDropout1D
from keras.layers.core import Activation, Dropout, Dense
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn import model_selection, metrics
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt


# Read in Review Datasets
reviews_train = pd.read_csv("goodreads_train.csv")
reviews_test = pd.read_csv("goodreads_test.csv")

# Removed unused categories such as book name, author name
reviews_train = reviews_train[["rating", "review_text"]]
reviews_test = reviews_test[["review_id", "review_text"]]

# Remove items with a rating of 0 since they will skew the results
#index = reviews_train[(reviews_train['rating'] == 0)].index
#reviews_train.drop(index , inplace=True)

# Undersample so we have same # of ratings in each category 1-5 to avoid bias for one rating category
#reviews_train = reviews_train.groupby('rating').head(28718)

# Filter the review text to make training more robust. Converts to lowercase, Removes symbols, Removes Double Spaces, punctuations, urls, numbers, emojis, and stopwords such as: a, the, we, I
reviews_train['review_text'] = reviews_train['review_text'].apply(lambda x: filter_text(x))
reviews_test['review_text'] = reviews_test['review_text'].apply(lambda x: filter_text(x))
train = reviews_train
#reviews_train.to_csv("reviews_train_filtered.csv", index=False)

# Split into Train and Validation: 70-30 split
reviews_train["split"] = reviews_train.apply(lambda x: "train" if random.randrange(0,100) > 10 else "valid", axis=1)
df_train = reviews_train[reviews_train["split"] == "train"]
df_val = reviews_train[reviews_train["split"] == "valid"]

# Generate tokens from review text datset, all words used in train dataset reviews are converted to numbers this creates a word-number dictionary
tokenizer=Tokenizer(oov_token="'oov'")
tokenizer.fit_on_texts(df_train['review_text'])
tokenizer_json = tokenizer.to_json()
with io.open('review_rating_tokenizer_v1.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))

# This is where the text to number conversion happens, we also set the max length to 200 words to speed up the processing
train_x = pad_sequences(tokenizer.texts_to_sequences(df_train['review_text']), maxlen=200)
val_x = pad_sequences(tokenizer.texts_to_sequences(df_val['review_text']), maxlen=200)
test_x = pad_sequences(tokenizer.texts_to_sequences(reviews_test['review_text']), maxlen=200)
train_y = df_train['rating']
val_y = df_val['rating']
# We have 5 classes or categories we need to train on
train_y_cat = to_categorical(df_train["rating"]-1, num_classes=6)
val_y_cat = to_categorical(df_val["rating"]-1, num_classes=6)

# Using GloVe Word Embeddings dictionary will allow us to us the context between words to draw more accurate conclusions
embeddings_dictionary = dict()
glove_file = open('glove.6B.100d.txt', encoding="utf8")
for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
glove_file.close()

# Apply Embedding Matrix to words in our tokenized dictionary.
max_words = len(tokenizer.word_index) + 1
embedding_matrix = np.zeros((max_words,100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index]=embedding_vector


# First Model Training Sequential (0.48667)
#model=Sequential()
#model.add(Embedding(max_words, 100, input_length=200, weights=[embedding_matrix], trainable=False))
#model.add(Bidirectional(LSTM(32))) #32
#model.add(Dropout(0.4))
#model.add(Dense(16, activation="relu"))#16
#model.add(Dense(5, activation="softmax"))#5
#model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['acc'])
#print(model.summary())
#model.fit(train_x, train_y_cat, epochs=20, batch_size=256, validation_data=(val_x, val_y_cat))
#pred = model.predict(val_x)
#model.save(f"rating_pred_seq_down.h5", save_format='h5')
#print(accuracy_score(val_y, [np.argmax(p)+1 for p in pred]))
#pred_new = model.predict(test_x)
#df_prediction_ratings = pd.DataFrame([np.argmax(pred)+1 for pred in pred_new], columns = ['rating'])
#df_review_id = pd.DataFrame(reviews_test['review_id'], columns = ['review_id'])
#dfx=pd.concat([df_review_id, df_prediction_ratings], axis=1)
#dfx.to_csv("submission.csv", sep=',', encoding='UTF-8', index=False)


# Model Training Sequential (0.56)
# This section defines are model structure, we use a sequential or linear model as the connection of our layers
# The Embedding layer will allow us to find context and relationships between words in review
# The Convolution layer works best applied after the embedding to find matches on trained patterns, very good at determining sentiment i.e positive or negative.
# In simple terms the convolution layer slides a filter across the text and forms groupings of words we can relationships between
# The max pooling layer is used after the convolutional ones to reduce over fitting and computational intensity on the feature maps from the convolutional layers
# The Bidirectional LSTM (Long Short Term Memory) layer is quite common for Recurrent Neural Networks to deal with vanishing and exploding gradients and holding relationships all the way from the beginning to end
# By making this layer bidirectional we can travel the layer both forward and backward for more meaningful relationships, this is particularly useful in sentences where we may need to back trace to determine the meaning an individual words has
# The dropout layer also reduces overfitting by deactivating certain nodes and paths the model may grow too dependent on during training
# The dense layers finally allow us to determine our final classifications and groupings 1-5, the rest of the code is just checking accuracy of the model
model=Sequential()
model.add(Embedding(max_words, 100, input_length=200, weights=[embedding_matrix], trainable=False))
model.add(Conv1D(filters=300, kernel_size=1, padding='same', activation='relu'))
model.add(Conv1D(filters=30, kernel_size=4, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Bidirectional(LSTM(32))) #32
model.add(Dropout(0.4))
model.add(Dense(16, activation="relu"))#16
model.add(Dense(6, activation="softmax"))#5
model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['acc'])
print(model.summary())
history = model.fit(train_x, train_y_cat, epochs=20, batch_size=256, validation_data=(val_x, val_y_cat))
pred = model.predict(val_x)
model.save(f"rating_pred_seq_down_3.h5", save_format='h5')
print(accuracy_score(val_y, [np.argmax(p)+1 for p in pred]))

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# Predictions
#df_prediction_ratings = pd.DataFrame([np.argmax(p)+1 for p in pred], columns = ['rating'])
#df_review_text = pd.DataFrame(reviews_train['review_text'], columns = ['review_text'])
#dfx=pd.concat([df_review_text, df_prediction_ratings], axis=1)
#dfx.to_csv("results_train.csv", sep=',', encoding='UTF-8', index=False)

#pred_new = model.predict(test_x)
#df_prediction_ratings = pd.DataFrame([np.argmax(pre)+1 for pre in pred_new], columns = ['rating'])
#df_review_id = pd.DataFrame(reviews_test['review_id'], columns = ['review_id'])
#dfx=pd.concat([df_review_id, df_prediction_ratings], axis=1)
#dfx.to_csv("submission4.csv", sep=',', encoding='UTF-8', index=False)

