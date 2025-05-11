import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.stats import chi2
import numpy as np
from datetime import datetime
import seaborn as sb

# 1. Collecte ou génération des données
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

csv =pd.read_csv("C:\\Users\\exe\\Desktop\\Scripts\\Coding\\Python\\Projects\\Traitement et Visualisation des données\\Achraf-Nizar\\flight.csv")

def Scatter(array1, array2, titre, xlabel, ylabel):
    plt.scatter(array1, array2, marker='o', color='blue', label='Données')
    plt.axhline(0, color='black', linewidth=1)  # Axe horizontal
    plt.axvline(0, color='black', linewidth=1)  # Axe vertical
    plt.title(titre)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    
def Histogramme(pd_data, bins, title, xlabel, ylabel):
    plt.hist(pd_data, bins=bins, edgecolor='black', alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.show()
    
def Boxplot(pd_data):
    plt.boxplot(pd_data, patch_artist=True, notch=True, vert=True, boxprops=dict(facecolor='lightblue', color='blue'), medianprops=dict(color='red'))
    plt.title("Boxplot")
    plt.ylabel("Values")
    plt.grid(True)
    plt.show()

def Heatmap(array, x_labels, y_labels):
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
    
def Barchart(pd_data, title, xlabel, ylabel):
    plt.bar(pd_data.index, pd_data.values, color='blue', alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    
def PieChart(pd_data, labels, title):
    plt.pie(pd_data, labels=pd_data.index, autopct='%1.1f%%', startangle=90)
    plt.title(title)
    plt.axis('equal')
    plt.show()

# Histogrammes
# Histogramme(csv["AGE"], 20, "", "", "")
# Histogramme(csv["FLIGHT_COUNT"], len(csv["FLIGHT_COUNT"].unique()), "", "", "")
# Histogramme(csv["SEG_KM_SUM"], 50, "", "", "")
# Histogramme(csv["Points_Sum"], 50, "", "", "")
# Histogramme(csv["AVG_INTERVAL"], 50, "", "", "")
# Histogramme(csv["SUM_YR_1"], 50, "", "", "")


CorrelationData = pd.concat([csv["FLIGHT_COUNT"], 
                             csv["SEG_KM_SUM"], 
                             csv["SUM_YR_1"], 
                             csv["SUM_YR_2"], 
                             csv["Points_Sum"], 
                             csv["AVG_INTERVAL"], 
                             csv["MAX_INTERVAL"], 
                             csv["EXCHANGE_COUNT"], 
                             csv["Point_NotFlight"]], axis=1)

# Calculate Correlation between each variable
# Correlation Heatmap

# Boxplots : 

# Boxplot(csv["FLIGHT_COUNT"])
# Boxplot(csv["GENDER"])

# Boxplot(csv["SEG_KM_SUM"])
# Boxplot(csv["FFP_TIER"])

# Boxplot(csv["Points_Sum"])
# Boxplot(csv["WORK_PROVINCE"])
# #or
# Boxplot(csv["WORK_COUNTRY"])

# Boxplot(csv["FFP_TIER"])
# Boxplot(csv["AVG_INTERVAL"])

# Scatter Plots :

# Scatter(csv["SEG_KM_SUM"], csv["FLIGHT_COUNT"], "", "", "")
# Scatter(csv["AVG_INTERVAL"], csv["FLIGHT_COUNT"], "", "", "")
# Scatter(csv["SUM_YR_1"], csv["SUM_YR_2"], "", "", "")
# Scatter(csv["avg_discount"], csv["Points_Sum"], "", "", "")

# Bar Charts :

# FFP_TierCount = pd.value_counts(csv["FFP_TIER"], sort=True)
# Barchart(FFP_TierCount, "", "", "")

# Moy_Points_Sum_by_Gender = pd.concat([csv["Points_Sum"], csv["GENDER"]], axis=1).groupby("GENDER")["Points_Sum"].mean().sort_values(ascending=False)
# Barchart(Moy_Points_Sum_by_Gender, "", "", "")

# Moy_Points_Sum_by_WorkCountry = pd.concat([csv["Points_Sum"], csv["WORK_COUNTRY"]], axis=1).groupby("WORK_COUNTRY")["Points_Sum"].mean().sort_values(ascending=False).head(20)
# Barchart(Moy_Points_Sum_by_WorkCountry, "", "", "")

# MembersCount_by_WORK_CITY = pd.value_counts(csv["WORK_CITY"], sort=True).head(20)
# Barchart(MembersCount_by_WORK_CITY, "", "", "")

# MembersCount_by_WORK_PROVINCE = pd.value_counts(csv["WORK_PROVINCE"], sort=True).head(20)
# Barchart(MembersCount_by_WORK_PROVINCE, "", "", "")



Quantitative = pd.concat([csv["FLIGHT_COUNT"], 
                             csv["SEG_KM_SUM"], 
                             csv["SUM_YR_1"], 
                             csv["AVG_INTERVAL"], 
                             csv["MAX_INTERVAL"], 
                             csv["EXCHANGE_COUNT"], 
                             csv["BP_SUM"],
                             csv["Points_Sum"]],axis=1)


# ACP
def ACP(dataMatrix):
    def AfficherGraphique(array1, array2, titre, xlabel, ylabel):
        """Afficher un graphique en nuage de points."""
        plt.scatter(array1, array2, marker='o', color='blue', label='Données')
        plt.axhline(0, color='black', linewidth=1)  # Axe horizontal
        plt.axvline(0, color='black', linewidth=1)  # Axe vertical
        plt.title(titre)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def AfficherMatrice(array, x_labels, y_labels):
        """Afficher une matrice sous forme de carte thermique avec annotations."""
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

    columns = dataMatrix.T.columns[0:]

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
        range(1, 15)
    )


ACP(Quantitative.iloc[:1000].T)
