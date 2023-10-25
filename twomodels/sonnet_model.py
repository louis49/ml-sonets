from keras.src.layers import Concatenate
from keras_tuner import HyperParameters

from data import Data
from keras import layers, Input, Model, optimizers, regularizers
from verset_model import VersetModel
from phon_model import PhonModel
import tensorflow as tf

class SonnetModel():
    def __init__(self, data: Data):
        self.data = data

    def build_model(self, hp):
        model = self.model(hp)
        learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log', default=0.001)
        model.compile(optimizer=optimizers.legacy.Adam(learning_rate=learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def model(self, hp):
        # Créer une instance de chaque modèle.
        phon_modeler = PhonModel(data=self.data)
        phon_model = phon_modeler.build_model(hp)

        verset_modeler = VersetModel(data=self.data)
        verset_model = verset_modeler.model(hp)

        # Définir les entrées.
        input_title = Input(shape=(self.data.title_max_size,), dtype="int32", name="title_input")
        input_phons = Input(shape=(self.data.phon_max_size * 14 + 1,), dtype="int32", name="phons_input")  # Les 14 rimes.
        input_verses = Input(shape=(self.data.text_max_size * 14 + 1,), dtype="int32", name="verses_input")  # Les 14 versets potentiels.

        # Traiter les entrées.
        processed_phons = input_phons[:, 1:]  # Supprimer le tag <start>.
        processed_verses = input_verses[:, 1:]  # Supprimer le tag <start>.

        # Utiliser le Modèle 1 pour obtenir les rimes encodées.
        encoded_phons = phon_model([input_title, processed_phons])
        encoded_phons_max = tf.argmax(encoded_phons, axis=-1)

        # Diviser les rimes encodées pour les utiliser individuellement.
        encoded_phons_split = tf.split(encoded_phons_max, num_or_size_splits=14, axis=1)
        encoded_verses_split = tf.split(processed_verses, num_or_size_splits=14, axis=1)

        # Utiliser le Modèle 2 pour générer des versets à partir de chaque rime.
        generated_verses = []
        for i in range(14):
            generated_verse = verset_model([input_title, encoded_phons_split[i], encoded_verses_split[i]])
            generated_verses.append(generated_verse)

        final_output = layers.Concatenate(axis=1)(generated_verses)

        model = Model(inputs=[input_title, input_phons, input_verses], outputs=final_output)

        return model

    def train(self, use_tuner=False):

        best_hyperparameters = HyperParameters()
        best_hyperparameters.Fixed('lstm_units', value=128)
        best_hyperparameters.Fixed('encoder_title_embedding_dim', value=128)
        best_hyperparameters.Fixed('encoder_phons_embedding_dim', value=128)
        best_hyperparameters.Fixed('decoder_embedding_dim', value=128)
        best_hyperparameters.Fixed('learning_rate', value=0.001)
        best_hyperparameters.Fixed('drop_out', value=0.38573)
        best_hyperparameters.Fixed('regularizer', value=0.0001)
        best_hyperparameters.Fixed('num_heads', value=5)

        model = self.build_model(best_hyperparameters)
