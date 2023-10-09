import numpy as np
import tensorflow as tf
from keras.src.layers import Conv2D, MaxPooling2D
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers.legacy import Adam

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = Sequential([
    Flatten(input_shape=(32, 32, 3)),
    Dense(128, activation='relu'),
    Dense(10)
])

#model = Sequential([
#    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
#    MaxPooling2D((2, 2)),
#    Conv2D(64, (3, 3), activation='relu'),
#    MaxPooling2D((2, 2)),
#    Conv2D(64, (3, 3), activation='relu'),

#    Flatten(),
#    Dense(64, activation='relu'),
#    Dense(100, activation='softmax')
#])

optimizer = Adam()
loss_fn = SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

factor = 100
# 100 * 10 * 10 images
for epoch in range(100 * factor):  # 10 époques
    imgs, labels = [], []
    for i in range(10):  # Pour chaque classe dans CIFAR-10
        idx = np.where(y_train == i)[0]
        selected_idx = np.random.choice(idx, 10, replace=False)  # Sélectionnez 100 images de chaque classe
        img, label = x_train[selected_idx], y_train[selected_idx]
        imgs.append(img)
        labels.append(label)
    imgs = np.concatenate(imgs, axis=0)
    labels = np.concatenate(labels, axis=0)
    model.train_on_batch(imgs, labels)
    _, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Époque {epoch + 1}, Accuracy: {acc * 100:.2f}%")
