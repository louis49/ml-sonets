import numpy as np
import tensorflow as tf
from keras.src.layers import Conv2D, MaxPooling2D, GaussianNoise, Dropout
from keras.src.regularizers import l2, l1_l2
from keras_tuner import BayesianOptimization
from scipy.constants import hp
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.optimizers.legacy import Nadam

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def build_model(hp):
    model = Sequential()

    model.add(GaussianNoise(0.1, input_shape=(32, 32, 3)))

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))

    filters = hp.Int('filters', min_value=32, max_value=512, step=32)

    model.add(Conv2D(filters=filters, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(filters=filters, kernel_size=(3, 3), activation='relu'))

    model.add(Flatten())
    model.add(Dense(units=hp.Int('dense_units', min_value=32, max_value=512, step=32),
                    activation='relu',
                    kernel_regularizer=l1_l2(l1=0.001, l2=0.001)))
    model.add(Dropout(0.3))
    model.add(Dense(units=10,
                    activation='softmax',
                    kernel_regularizer=l1_l2(l1=0.001, l2=0.001)))

    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(
                  hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')), #values=[1e-2, 1e-3, 1e-4]
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
    project_name='intro_to_kt',
    executions_per_trial=2,

    overwrite=False
)

tuner.search(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Obtenir le meilleur modèle
best_model = tuner.get_best_models(num_models=1)[0]

# Afficher un résumé des résultats
tuner.results_summary()
