import pandas as pd
import matplotlib.pyplot as plt

# Charger les données depuis un fichier CSV
data = pd.read_csv('test.csv', delimiter=';')  # Remplacez par le chemin réel du fichier CSV

# Définir les couleurs selon la troisième colonne
colors = ['red' if val == 0 else 'green' for val in data['RES']]

# Créer un graphique en nuage de points
plt.scatter(data['LSTM'], data['D_DIM'], c=colors, s=1)

# Ajouter des titres et des étiquettes
plt.title('Graphique en nuage de points avec couleurs conditionnelles')
plt.xlabel('LSTM')
plt.ylabel('D_DIM')

# Afficher le graphique
plt.show()
print()
