import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

# Variable globale pour activer/désactiver l'affichage des graphiques
pltShow = True

# ----------------------------- Fonctions Utilitaires -----------------------------

def AfficherGraphique(array1, array2, titre, xlabel, ylabel):
    """Afficher un graphique en nuage de points."""
    if pltShow:
        plt.scatter(array1, array2, marker='o', color='blue', label='Données')
        plt.axhline(0, color='black', linewidth=1)  # Axe horizontal
        plt.axvline(0, color='black', linewidth=1)  # Axe vertical
        plt.title(titre)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

def AfficherMatrice(array, x_labels, y_labels):
    """Afficher une matrice sous forme de carte thermique avec annotations."""
    if pltShow:
        fig, ax = plt.subplots()
        cax = ax.matshow(array, cmap='coolwarm')
        for (i, j), val in np.ndenumerate(array):
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black')
        fig.colorbar(cax)
        ax.set_xticks(range(len(x_labels)))
        ax.set_yticks(range(len(y_labels)))
        ax.set_xticklabels(x_labels, rotation=90)
        ax.set_yticklabels(y_labels)
        plt.show()

# ----------------------------- Chargement des Données -----------------------------

# Charger les données depuis un fichier Excel
dataMatrix = pd.read_excel(
    "C:\\Users\\exe\\Desktop\\Scripts\\Coding\\Python\\Projects\\Data Analyst\\ACP\\nobel.xlsx"
).T

columns = dataMatrix.T.columns[1:]

# Supprimer des lignes spécifiques
dataMatrix_CR = np.delete(dataMatrix, 0, axis=0)

# ----------------------------- Prétraitement --------------------------------------

# Normaliser les variables (centrage et réduction)
for variable in dataMatrix_CR:
    mean = np.average(variable)
    std_dev = np.std(variable)
    for i in range(len(variable)):
        variable[i] = (variable[i] - mean) / std_dev
        

# ----------------------------- Matrice de Corrélation -----------------------------

# Calculer la matrice de corrélation
Mat_de_Correlation = (dataMatrix_CR @ dataMatrix_CR.T) / len(dataMatrix_CR[0])
Mat_de_Correlation = Mat_de_Correlation.astype(float)

# Afficher la matrice de corrélation
AfficherMatrice(
    Mat_de_Correlation,
    columns,
    columns
)

# ----------------------------- Valeurs Propres et Inertie -------------------------

# Calculer les valeurs propres
valPropres = np.linalg.eigvals(Mat_de_Correlation).real.astype(float).tolist()

# Indexer les valeurs propres
valpropre_indexed = [[i, val] for i, val in enumerate(valPropres)]
print("Valeurs propres : ", valpropre_indexed, "\n")

# Calculer l'inertie expliquée
inertie_explique = [[index, val * 100 / sum(valPropres)] for index, val in valpropre_indexed]
inertie_explique.sort(key=lambda x: x[1], reverse=True)
print("Inerties expliquées : ", inertie_explique, "\n")

# Calculer l'inertie cumulée
inertie_cumule = []
cumulative = 0
for index, val in inertie_explique:
    cumulative += val
    inertie_cumule.append([index, cumulative])
print("Inerties Cumulées : ", inertie_cumule, "\n")

# ----------------------------- Composantes Principales ----------------------------

# Identifier les deux plus grandes valeurs propres
landa1 = max(valPropres)
valPropres.remove(landa1)
landa2 = max(valPropres)

# Calculer les vecteurs propres
eig = np.linalg.eig(Mat_de_Correlation)
v1 = eig[1][:, inertie_explique[0][0]]
v2 = eig[1][:, inertie_explique[1][0]]

# Projeter les données sur les deux premières composantes principales
vector1 = dataMatrix_CR.T @ v1
vector2 = dataMatrix_CR.T @ v2

# Afficher le nuage des individus
AfficherGraphique(
    vector1,
    vector2,
    "Nuage des individus",
    f"F1({inertie_explique[0][1]}%)",
    f"F2({inertie_explique[1][1]}%)"
)

# ----------------------------- Cercle de Corrélation ------------------------------

def CalculerCercleCorrelation():
    """Calculer les coordonnées du cercle de corrélation."""
    cercle_correlation = []
    for i in range(len(dataMatrix_CR)):
        temp = []
        for j in range(len(dataMatrix_CR)):
            temp.append(eig[1][i][j] * math.sqrt(eig[0][j]))
        cercle_correlation.append(temp)
    return np.array(cercle_correlation, dtype=float)

cercle_correlation = CalculerCercleCorrelation()

AfficherMatrice(cercle_correlation, ["F"+str(i) for i in range(1, len(cercle_correlation)+1)], columns)

# Afficher le cercle de corrélation
def AfficherCercleCorrelation():
    fig, ax = plt.subplots()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='black', linewidth=1)
    circle = plt.Circle((0, 0), 1, color='blue', fill=False)
    ax.add_artist(circle)
    plt.title("Cercle de corrélation")
    plt.xlabel("F1")
    plt.ylabel("F2")
    plt.grid()
    for i in range(len(cercle_correlation)):
        x, y = cercle_correlation[i][0], cercle_correlation[i][1]
        plt.quiver(0, 0, x, y, angles='xy', scale_units='xy', scale=1, color='blue', alpha=0.8)
        plt.text(x, y, columns[i], fontsize=8, ha='right', va='bottom')
    plt.show()

AfficherCercleCorrelation()

# ----------------------------- Qualité de Représentation --------------------------

def CalculerQualiteRepresentation():
    """Calculer la qualité de représentation des individus."""
    qualite_representation = []
    for i in range(dataMatrix_CR.T.shape[0]):
        somme_carres = np.linalg.norm(dataMatrix_CR.T[i], 2) ** 2
        q1 = (vector1[i] ** 2) / somme_carres
        q2 = (vector2[i] ** 2) / somme_carres
        qualite_representation.append([q1, q2, q1 + q2])
    return qualite_representation

# Afficher la matrice de qualité de représentation
AfficherMatrice(
    CalculerQualiteRepresentation(),
    ["F1", "F2", "Total"],
    range(1, dataMatrix.T.shape[0] + 1)
)
