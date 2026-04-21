from keras.callbacks import ModelCheckpoint, EarlyStopping


def save_best_model(path:str):

    return ModelCheckpoint(path,
                           monitor='categorical_accuracy',
                           save_best_only=True,
                           mode='max')

def early_stop():

    return EarlyStopping(
        monitor='categorical_accuracy',
        patience=50,
        restore_best_weights=True
    )