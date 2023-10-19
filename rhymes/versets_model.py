import keras
from tensorflow.keras import layers


class VersetsModel():
    def __init__(self, title_words, text_words, phon_words, max_len_title, max_len_text, max_len_phon):
        self.title_words = title_words
        self.text_words = text_words
        self.phon_words = phon_words
        self.title_max_len = max_len_title
        self.text_max_len = max_len_text
        self.phon_max_len = max_len_phon

    def build_model(self, hp, ):



        #drop_out = hp.Float('drop_out', min_value=0.0, max_value=1.0, step=0.05, default=0.38)
        #regularizer = hp.Float('regularizer', min_value=1e-5, max_value=1e-2, sampling='log', default=0.0001)
        #num_heads = hp.Int("num_heads", min_value=3, max_value=24, step=1, default=12)

        embedding_dim = hp.Int("embedding_dim", min_value=8, max_value=512, step=8, default=184)
        lstm_units = hp.Int("lstm_units", min_value=8, max_value=512, step=8, default=168)
        learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log', default=0.001)

        # Entrées
        title_input = keras.Input(shape=(self.title_max_len,), dtype="int32", name="title")
        phon_input = keras.Input(shape=(self.phon_max_len,), dtype="int32", name="rhyme")
        decoder_input = keras.Input(shape=(self.text_max_len-1,), dtype="int32", name="decoder_input")

        # Embeddings
        title_embedding_layer = layers.Embedding(self.title_words, embedding_dim)  # title_words est le vocabulaire pour les titres
        phon_embedding_layer = layers.Embedding(self.phon_words, embedding_dim)  # total_words est le vocabulaire total
        title_embedded = title_embedding_layer(title_input)
        phon_embedded = phon_embedding_layer(phon_input)

        # LSTM pour les entrées
        title_lstm = layers.LSTM(lstm_units, return_sequences=True, return_state=True)
        phon_lstm = layers.LSTM(lstm_units, return_sequences=True, return_state=True)

        title_outputs, title_state_h, title_state_c = title_lstm(title_embedded)
        phon_outputs, phon_state_h, phon_state_c = phon_lstm(phon_embedded)

        # Vous pouvez éventuellement utiliser l'état caché (state_h) des deux LSTMs pour un contexte combiné
        combined_state_h = layers.Concatenate()([title_state_h, phon_state_h])
        combined_state_c = layers.Concatenate()([title_state_c, phon_state_c])

        # Décodeur sans attention
        decoder_embedding_layer = layers.Embedding(self.text_words, embedding_dim)
        decoder_embedded = decoder_embedding_layer(decoder_input)

        # Notez que nous utilisons les états combinés ici directement.
        decoder_lstm = layers.LSTM(2 * lstm_units, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_embedded, initial_state=[combined_state_h, combined_state_c])

        # Directement à la sortie sans attention.
        output_layer = layers.TimeDistributed(layers.Dense(self.text_words+1, activation='softmax'))
        final_outputs = output_layer(decoder_outputs)  # Ici, nous passons directement les sorties du décodeur.

        # Création et compilation du modèle restent les mêmes.
        model = keras.Model(inputs=[title_input, phon_input, decoder_input], outputs=final_outputs)

        # Compilation du modèle (à ajuster selon vos besoins)
        model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

        return model
