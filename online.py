import numpy as np
import tensorflow as tf
from keras.src.layers import Conv2D, MaxPooling2D, GaussianNoise, Dropout
from keras.src.regularizers import l2, l1_l2
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.optimizers.legacy import Nadam

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#model = Sequential([
#    GaussianNoise(0.1, input_shape=(32, 32, 3)),
#    Flatten(),
#    Dense(96, activation='relu', kernel_regularizer=l1_l2(l1=0.001, l2=0.001)),
#    Dropout(0.3),
#    Dense(10, kernel_regularizer=l1_l2(l1=0.001, l2=0.001))
#])

model = Sequential([
    GaussianNoise(0.1, input_shape=(32, 32, 3)),
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(160, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(160, (3, 3), activation='relu'),

    Flatten(),
    Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=0.001, l2=0.001)),
    Dropout(0.3),
    Dense(10, activation='softmax', kernel_regularizer=l1_l2(l1=0.001, l2=0.001))
])

optimizer = Adam(learning_rate=0.001) #lr=0.001
loss_fn = SparseCategoricalCrossentropy(from_logits=False)
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

accumulated_gradients = [tf.zeros_like(var) for var in
                         model.trainable_variables]  # Initialiser les gradients accumulés à zéro

factor = 100
for epoch in range(1000 * factor):  # 5000 époques
    idx = np.random.choice(len(x_train), 1000, replace=False)  # Sélectionner 2 images aléatoirement à chaque époque
    img, label = x_train[idx], y_train[idx]

    with tf.GradientTape() as tape:
        predictions = model(img, training=True)
        loss = loss_fn(label, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    accumulated_gradients = [acc_grad + grad for acc_grad, grad in zip(accumulated_gradients, gradients)]

    if (epoch + 1) % 1 == 0:  # Appliquer les gradients accumulés tous les 50 époques
        optimizer.apply_gradients(zip(accumulated_gradients, model.trainable_variables))
        accumulated_gradients = [tf.zeros_like(var) for var in
                                 model.trainable_variables]  # Réinitialiser les gradients accumulés
        _, acc = model.evaluate(x_test, y_test, verbose=0)
        print(f"Époque {epoch + 1}, Accuracy: {acc * 100:.2f}%")
