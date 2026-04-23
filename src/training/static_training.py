from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, Reshape, Conv1D, Flatten
import numpy as np
import os

def train_static_Dense_NN(X_TRAIN, Y_TRAIN, actions, epochs, callback):

    model = Sequential()

    model.add(Input(shape=(126,)))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(len(actions), activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    model.fit(X_TRAIN, Y_TRAIN, epochs=epochs, callbacks=callback)

    os.makedirs("models", exist_ok=True)

    return model


def train_static_Conv1D_NN(X_TRAIN, Y_TRAIN, actions, epochs, callback):
    model = Sequential()

    model.add(Input(shape=(126,)))
    model.add(Reshape((42, 3)))
    model.add(Conv1D(64, activation='relu', kernel_size=3))
    model.add(Conv1D(128, activation='relu', kernel_size=3))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(len(actions), activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    model.fit(X_TRAIN, Y_TRAIN, epochs=epochs, callbacks=callback)

    
    os.makedirs("models", exist_ok=True)

    return model