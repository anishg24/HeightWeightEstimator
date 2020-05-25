import os
from pathlib import Path

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import RMSprop

height_dir, weight_dir = [os.path.join(str(Path(os.getcwd()).parent) + f"/data/{file}") for file in
                          ["height.npy", "weight.npy"]]
heights, weights = np.load(height_dir), np.load(weight_dir)

# train_h, train_w, test_h, test_w = train_test_split(height_dir, height_dir, test_size=0.2)


def build_model():
    model = Sequential()
    model.add(Dense(64, activation="relu", input_shape=[1]))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1))

    optimizer = RMSprop(0.001)

    model.compile(loss="mse", optimizer=optimizer, metrics=["mae", "mse"])
    return model


model = build_model()

EPOCHS = 1000
history = model.fit(heights, weights, epochs=EPOCHS, validation_split=0.2, verbose=0)
