# ===================================================================================================
# PROBLEM B4
#
# Build and train a classifier for the BBC-text dataset.
# This is a multiclass classification problem.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in: http://mlg.ucd.ie/datasets/bbc.html.
#
# Desired accuracy and validation_accuracy > 91%
# ===================================================================================================

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd
import numpy as np


def solution_B4():
    bbc = pd.read_csv(
        'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/bbc-text.csv')

    # DO NOT CHANGE THIS CODE
    # Make sure you used all of these parameters or you can not pass this test
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_portion = .8

    # YOUR CODE HERE
    # Using "shuffle=False"
    labels = bbc["category"].values.tolist()
    sentences = bbc["text"].values.tolist()

    training_size = int(len(sentences) * training_portion)
    training_sentences = sentences[:training_size]
    training_labels = labels[:training_size]
    validation_sentences = sentences[training_size:]
    validation_labels = labels[training_size:]

    # Fit your tokenizer with training data
    tokenizer =  Tokenizer(num_words = vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index

    training_sequences = tokenizer.texts_to_sequences(training_sentences)
    training_padded_sequences = pad_sequences(training_sequences, padding=padding_type, maxlen=max_length, truncating=trunc_type)
    validation_sequences = tokenizer.texts_to_sequences(validation_sentences)
    validation_padded_sequences = pad_sequences(validation_sequences, padding=padding_type, maxlen=max_length, truncating=trunc_type)

    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(labels)
    label_word_index = label_tokenizer.word_index
    training_label_sequences = label_tokenizer.texts_to_sequences(training_labels)
    training_label_sequences = np.array(training_label_sequences)
    validation_label_sequences = label_tokenizer.texts_to_sequences(validation_labels)
    validation_label_sequences = np.array(validation_label_sequences)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv1D(64, 5, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=4),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(6, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    model.fit(
        training_padded_sequences,
        training_label_sequences,
        epochs=100,
        validation_data=(
            validation_padded_sequences,
            validation_label_sequences),
        verbose=2)

    return model

    # The code below is to save your model as a .h5 file.
    # It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_B4()
    model.save("model_B4.h5")
