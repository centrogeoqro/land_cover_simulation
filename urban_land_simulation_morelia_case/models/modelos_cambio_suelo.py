import numpy as np
from imblearn.under_sampling import RandomUnderSampler
undersampler = RandomUnderSampler()

def MLP_model(xtrain, ytrain, xvalid, yvalid):
    import tensorflow as tf
    X_train_resampled, y_train_resampled = undersampler.fit_resample(xtrain, ytrain)
    inputs = tf.keras.Input(shape=(np.shape(X_train_resampled)[1],), name='input')

    x = tf.keras.layers.Dense(150, activation='sigmoid')(inputs)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(100, activation='sigmoid')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(50, activation='sigmoid')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    callback = tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = 1)

    MLP = tf.keras.Model(inputs=inputs, outputs=outputs)

    MLP.compile(optimizer = 'adam',
                loss = 'binary_crossentropy',
                #loss = CustomAccuracy(),
                #metrics = 'binary_crossentropy'
                metrics = 'binary_crossentropy'
                )

    history_mlp = MLP.fit(X_train_resampled, y_train_resampled, validation_data=(xvalid, yvalid),  callbacks=[callback] , epochs = 40, batch_size = int(len(y_train_resampled)*0.8))
    return history_mlp, MLP


def perceptron_model(xtrain, ytrain, xvalid, yvalid):
    import tensorflow as tf

    X_train_resampled, y_train_resampled = undersampler.fit_resample(xtrain, ytrain)
    inputs = tf.keras.Input(shape=(np.shape(X_train_resampled)[1],), name='input')


    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(inputs)

    callback = tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = 20)

    perceptron = tf.keras.Model(inputs=inputs, outputs=outputs)

    perceptron.compile(optimizer = 'adam',
                loss = 'binary_crossentropy',
                #loss = CustomAccuracy(),
                #metrics = 'binary_crossentropy'
                metrics = 'binary_crossentropy'
                )

    history_perceptron = perceptron.fit(X_train_resampled, y_train_resampled,  callbacks=[callback] , epochs = 10000, batch_size = int(len(y_train_resampled)*1))
    return history_perceptron, perceptron