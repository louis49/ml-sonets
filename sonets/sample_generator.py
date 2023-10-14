from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.callbacks import Callback
import numpy as np
import re

class GenerateSample(Callback):

    def __init__(self, input_text, tokenizer, total_words, max_decoder_seq_length=100, max_encoder_seq_length=100, every_epoch=1):
        super(GenerateSample, self).__init__()
        self.input_text = input_text
        self.tokenizer = tokenizer
        self.total_words = total_words
        self.max_decoder_seq_length = max_decoder_seq_length
        self.max_encoder_seq_length = max_encoder_seq_length
        self.every_epoch = every_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.every_epoch == 0 and epoch > 0:
            generated_text = self.generate_text(self.model)
            print("\nSample generation at epoch {}: \n{}".format(epoch + 1, generated_text))

    def generate_text(self, model):
        curr_seq = self.tokenizer.texts_to_sequences([self.input_text])[0]
        generated_text = [self.tokenizer.index_word[idx] for idx in curr_seq]

        i = len(curr_seq)
        while i < self.max_decoder_seq_length:
            encoder_input_data = pad_sequences([curr_seq], maxlen=self.max_decoder_seq_length, padding='post')
            decoder_input_data = pad_sequences([curr_seq], maxlen=self.max_decoder_seq_length, padding='post')

            # Prédiction du mot suivant
            predictions = model.predict([encoder_input_data, decoder_input_data], verbose=False)
            next_word_idx = np.argmax(predictions[0], axis=-1)

            if next_word_idx[i] == 0:# Si c'est un padding, on continue
                progress_percent = (i / self.max_decoder_seq_length) * 100
                print(f"\rGenerating progress: {progress_percent:.2f}%", end="")
                i += 1
                continue

            curr_seq.append(next_word_idx[i])
            next_word = self.tokenizer.index_word[next_word_idx[i]]
            generated_text.append(next_word)

            # Si le token '</sonnet>' est atteint, arrêter la prédiction
            if next_word == '</sonnet>':
                break

            progress_percent = (i / self.max_decoder_seq_length) * 100
            print(f"\rGenerating progress: {progress_percent:.2f}%", end="")

            i += 1
        return self.format_sonnet(' '.join(generated_text))

    def format_sonnet(self, sonnet):
        # Supprimer les balises <sonnet>, <title>, </title>, </sonnet>
        sonnet = re.sub(r'<sonnet>|<title>|</title>|</sonnet>', '', sonnet)

        # Supprimer les espaces en début de texte
        sonnet = re.sub(r'^ +', '', sonnet)

        # Remplacer les balises <strophe[A-D]> et <line[A-D]> par un saut de ligne
        sonnet = re.sub(r'<strophe[a-d]>|<line[a-d]>', '\n', sonnet)

        # Supprimer les balises de fermeture </strophe[A-D]> et </line[A-D]>
        sonnet = re.sub(r'</strophe[a-d]>|</line[a-d]>', '', sonnet)

        # Supprimer tout le contenu entre les balises <rime> et </rime>, ainsi que les balises elles-mêmes
        sonnet = re.sub(r'<rime>.*?</rime>', '', sonnet)

        # Supprimer les doubles espaces et les espaces en début de texte
        sonnet = re.sub(r' +|^ +', ' ', sonnet)

        # Supprimer les espaces après une apostrophe
        sonnet = re.sub(r"' +", "'", sonnet)

        # Supprimer les espaces qui précèdent une virgule ou un point
        sonnet = re.sub(r" +([,.])", r"\1", sonnet)

        # Supprimer les espaces qui suivent un saut de ligne
        sonnet = re.sub(r"\n +", "\n", sonnet)

        return sonnet