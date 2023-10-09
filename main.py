import tensorflow as tf
import numpy as np
from PIL import ImageEnhance, Image
from tensorflow.keras.datasets import cifar10

(x_train, _), (x_test, _) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

def modify_hsv(image_pil, h_factor=1.0, s_factor=1.0, v_factor=1.0):
    img_hsv = image_pil.convert('HSV')
    arr_hsv = np.array(img_hsv).astype(int)
    arr_hsv[..., 0] = (arr_hsv[..., 0] * h_factor) % 256
    arr_hsv[..., 1] = np.clip(arr_hsv[..., 1] * s_factor, 0, 255)
    arr_hsv[..., 2] = np.clip(arr_hsv[..., 2] * v_factor, 0, 255)
    img_hsv = Image.fromarray(arr_hsv.astype('uint8'), 'HSV').convert('RGB')
    return img_hsv

def apply_gamma_correction(image, gamma):
    image = np.array(image)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    corrected_image = np.take(table, image)
    return Image.fromarray(np.uint8(corrected_image))
def custom_preprocess():
    random_indices = np.random.choice(50000, 1000, replace=False)
    train_data = []
    train_labels = []
    for idx in random_indices:
        img = x_train[idx]
        img_pil = Image.fromarray((img * 255).astype('uint8'))
        color_val = np.random.normal(0.5, 0.1)
        contrast_val = np.random.normal(1, 0.1)
        brightness_val = np.random.normal(1, 0.1)
        sharpness_val = np.random.normal(1, 0.1)
        gamma_val = np.random.normal(1, 0.1)
        h_factor = np.random.uniform(0, 1)
        s_factor = np.random.uniform(0, 1)
        v_factor = np.random.uniform(0, 1)

        enhancer = ImageEnhance.Color(img_pil)
        img_pil = enhancer.enhance(color_val)

        enhancer = ImageEnhance.Contrast(img_pil)
        img_pil = enhancer.enhance(contrast_val)

        enhancer = ImageEnhance.Brightness(img_pil)
        img_pil = enhancer.enhance(brightness_val)

        enhancer = ImageEnhance.Sharpness(img_pil)
        img_pil = enhancer.enhance(sharpness_val)

        img_pil = apply_gamma_correction(img_pil, gamma_val)

        img_pil = modify_hsv(img_pil, h_factor, s_factor, v_factor)

        modified_img = np.array(img_pil) / 255.0
        train_data.append((img, modified_img))
        train_labels.append([color_val, contrast_val, brightness_val, sharpness_val, gamma_val, h_factor, s_factor, v_factor])

    train_data = np.array(train_data, dtype='float32')
    train_labels = np.array(train_labels, dtype='float32')

    return train_data, train_labels

class AdvancedCNNModel(tf.keras.Model):
    def __init__(self):
        super(AdvancedCNNModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.fc1 = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4))
        self.fc2 = tf.keras.layers.Dense(8, activation='linear')

    def call(self, img_data_batch):
        norm_img_batch, enh_img_batch = img_data_batch[:, 0], img_data_batch[:, 1]

        #print(f"Norm Img Batch Shape: {norm_img_batch.shape}")
        #print(f"Enh Img Batch Shape: {enh_img_batch.shape}")

        x = tf.concat([norm_img_batch, enh_img_batch], axis=-1)

        #print(f"After concat: {x.shape}")

        x = self.conv1(x)

        #print(f"After conv1: {x.shape}")

        x = self.conv2(x)

        #print(f"After conv2: {x.shape}")

        x = tf.keras.layers.Flatten()(x)

        #print(f"After flatten: {x.shape}")

        x = self.fc1(x)

        #print(f"After fc1: {x.shape}")

        x = self.fc2(x)

        #print(f"After fc2: {x.shape}")

        return x


def custom_accuracy(y_true, y_pred, tolerance=0.01):
    correct_count = tf.reduce_sum(tf.cast(tf.abs(y_true - y_pred) < tolerance, tf.float32))
    total_count = tf.size(y_true, out_type=tf.float32)
    return (correct_count / total_count) * 100


model = AdvancedCNNModel()
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.legacy.Adam()

for epoch in range(10000):
    total_accuracy = 0
    num_batches = 0

    train_data, train_labels = custom_preprocess()
    train_data_np = np.array([td for td in train_data], dtype='float32')
    train_labels_np = np.array(train_labels, dtype='float32')

    batch_size = 50
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data_np, train_labels_np)).batch(batch_size)
    val_dataset = train_dataset.take(1)
    train_dataset = train_dataset.skip(1)

    for (img_data_batch, true_vals) in train_dataset:
        with tf.GradientTape() as tape:
            pred_vals = model(img_data_batch, training=True)
            loss = loss_fn(true_vals, pred_vals)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        total_accuracy += custom_accuracy(true_vals, pred_vals)
        num_batches += 1

    avg_train_accuracy = total_accuracy / num_batches

    total_val_accuracy = 0
    num_val_batches = 0
    for (img_data_batch, true_vals) in val_dataset:
        pred_vals = model(img_data_batch)
        val_loss = loss_fn(true_vals, pred_vals)

        total_val_accuracy += custom_accuracy(true_vals, pred_vals)
        num_val_batches += 1

    avg_val_accuracy = total_val_accuracy / num_val_batches

    print(
        f"Epoch {epoch + 1}, Loss: {loss.numpy()}, Train Accuracy: {avg_train_accuracy}%, Val Loss: {val_loss.numpy()}, Val Accuracy: {avg_val_accuracy}%")
