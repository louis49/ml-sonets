import pandas as pd
import matplotlib.pyplot as plt

# Charger les données depuis un fichier CSV
data = pd.read_csv('hparams_table.csv', delimiter=',')  # Remplacez par le chemin réel du fichier CSV

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#colors = ['red' if float(val) < 0.3 else 'orange' if float(val) < 0.826 else 'green' for val in data['evaluation_last_word_vs_iterations']]
colors = ['red' if float(val) < 0.3 else 'green' for val in data['evaluation_last_word_vs_iterations']]

ax.scatter(data['embedding_dim_decoder_input'], data['embedding_dim_encoder_input'], data['lstm_units'], c=data['evaluation_last_word_vs_iterations'], cmap=plt.cm.RdYlGn, s=20)  # s est la taille

ax.set_xlabel('dim_decoder_input')
ax.set_ylabel('dim_encoder_input')
ax.set_zlabel('lstm_units')

plt.show()
