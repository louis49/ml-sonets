import numpy as np
import tensorflow as tf
from keras_tuner.tuners import BayesianOptimization
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from kerastuner.tuners import Hyperband

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def build_model(hp):
    model = Sequential()

    # Pour le modèle à couches denses
    model.add(Flatten(input_shape=(32, 32, 3)))
    model.add(Dense(units=hp.Int('units', min_value=90, max_value=128, step=1), #96
                    activation='relu'))
    #hp.Choice('activation', values=['relu', 'tanh', 'sigmoid', 'swish'])
    model.add(Dense(10))

    # Décommentez ci-dessous pour le modèle CNN
    # model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(Flatten())
    # model.add(Dense(64, activation='relu'))
    # model.add(Dense(10, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(
                  hp.Float('learning_rate', min_value=1e-4, max_value=1e-3, sampling='log')), #values=[1e-2, 1e-3, 1e-4]
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

tuner = BayesianOptimization( #Hyperband
    build_model,
    objective='val_accuracy',
    max_trials=100,  # nombre total d'essais
    num_initial_points=10,  # Combien d'essais sont utilisés pour initialiser les modèles avant de basculer vers la recherche bayésienne
    #max_epochs=10,
    directory='my_dir',
    project_name='intro_to_kt'
)

tuner.search(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Obtenir le meilleur modèle
best_model = tuner.get_best_models(num_models=1)[0]

# Afficher un résumé des résultats
tuner.results_summary()
