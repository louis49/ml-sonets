from tensorflow.keras import layers
from tensorflow import keras

class SonnetModel():
    def __init__(self, input_dim, output_dim, total_words):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.total_words = total_words

    def build_model(self, hp):
        LSTM_MAX = 512
        DIM_MAX = 512
        HEAD_MAX = 8

        lstm_units = hp.Int("lstm_units", min_value=8, max_value=LSTM_MAX, step=8, default=LSTM_MAX)  # 8-128
        embedding_dim = hp.Int("embedding_dim", min_value=8, max_value=DIM_MAX, step=8, default=DIM_MAX)  # 8-128
        learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')
        drop_out = hp.Float('drop_out', min_value=0.0, max_value=1.0, default=0.5)
        regularizer = hp.Float('regularizer', min_value=1e-5, max_value=1e-2, sampling='log')
        num_heads = hp.Int("num_heads", min_value=1, max_value=HEAD_MAX, step=1, default=HEAD_MAX)

        encoder_input = keras.Input(shape=(self.input_dim,))
        encoder_embedding = layers.Embedding(
            self.total_words,
            embedding_dim
        )(encoder_input)
        encoder_embedding = layers.LayerNormalization()(encoder_embedding)
        encoder_embedding = layers.Dropout(drop_out)(encoder_embedding)
        encoder_output, forward_h, forward_c, backward_h, backward_c = layers.Bidirectional(layers.LSTM(
            lstm_units,
            return_state=True,
            return_sequences=True,
            # dropout=DROP_OUT,
            # recurrent_dropout=DROP_OUT,
            kernel_regularizer=keras.regularizers.l2(regularizer),
            recurrent_regularizer=keras.regularizers.l2(regularizer),
            bias_regularizer=keras.regularizers.l2(regularizer)
        ))(encoder_embedding)
        state_h = layers.Concatenate()([forward_h, backward_h])
        state_c = layers.Concatenate()([forward_c, backward_c])
        encoder_states = [state_h, state_c]

        encoder_attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)
        encoder_attention_output = encoder_attention(query=encoder_output, key=encoder_output, value=encoder_output)
        encoder_output = layers.Concatenate(axis=-1)([encoder_output, encoder_attention_output])

        # Décodeur
        decoder_input = keras.Input(shape=(self.output_dim,))
        decoder_embedding = layers.Embedding(self.total_words, embedding_dim)(decoder_input)
        decoder_embedding = layers.LayerNormalization()(decoder_embedding)
        decoder_embedding = layers.Dropout(drop_out)(decoder_embedding)
        decoder_output, _, _ = layers.LSTM(
            2 * lstm_units,
            return_sequences=True,
            return_state=True,
            # dropout=DROP_OUT,
            # recurrent_dropout=DROP_OUT,
            kernel_regularizer=keras.regularizers.l2(regularizer),
            recurrent_regularizer=keras.regularizers.l2(regularizer),
            bias_regularizer=keras.regularizers.l2(regularizer)
        )(decoder_embedding, initial_state=encoder_states)

        attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)
        attention_output = attention(query=decoder_output, key=encoder_output, value=encoder_output)

        decoder_concat = layers.Concatenate(axis=-1)([decoder_output, attention_output])

        decoder_dense = layers.TimeDistributed(layers.Dense(self.total_words, activation='softmax'))
        decoder_output = decoder_dense(decoder_concat)

        model = keras.Model([encoder_input, decoder_input], decoder_output)

        model.compile(
            optimizer=keras.optimizers.legacy.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model