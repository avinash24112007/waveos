from keras.models import Sequential
from keras.layers import LSTM, Dense
import os
import numpy as np


def train_action_NN(X_Train,
                    Y_Train, 
                    epochs,
                    callback:list,
                    actions):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape = (30, 126)))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(np.array(actions)[0], activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    model.fit(X_Train, Y_Train, epochs=epochs, callbacks=callback)

    os.makedirs("models", exist_ok=True)
    model.save(os.path.join("models", "action_model.keras"))






