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

csv =pd.read_csv("C:\\Users\\achraf\\Documents\\visDeDonnee\\TheFetcher\\flight.csv")

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
Histogramme(csv["AGE"], 20, "", "", "")
Histogramme(csv["FLIGHT_COUNT"], len(csv["FLIGHT_COUNT"].unique()), "", "", "")
Histogramme(csv["SEG_KM_SUM"], 50, "", "", "")
Histogramme(csv["Points_Sum"], 50, "", "", "")
Histogramme(csv["AVG_INTERVAL"], 50, "", "", "")
Histogramme(csv["SUM_YR_1"], 50, "", "", "")


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

Scatter(csv["SEG_KM_SUM"], csv["FLIGHT_COUNT"], "", "", "")
Scatter(csv["AVG_INTERVAL"], csv["FLIGHT_COUNT"], "", "", "")
Scatter(csv["SUM_YR_1"], csv["SUM_YR_2"], "", "", "")
Scatter(csv["avg_discount"], csv["Points_Sum"], "", "", "")

# Bar Charts :

FFP_TierCount = pd.value_counts(csv["FFP_TIER"], sort=True)
Barchart(FFP_TierCount, "", "", "")

Moy_Points_Sum_by_Gender = pd.concat([csv["Points_Sum"], csv["GENDER"]], axis=1).groupby("GENDER")["Points_Sum"].mean().sort_values(ascending=False)
Barchart(Moy_Points_Sum_by_Gender, "", "", "")

Moy_Points_Sum_by_WorkCountry = pd.concat([csv["Points_Sum"], csv["WORK_COUNTRY"]], axis=1).groupby("WORK_COUNTRY")["Points_Sum"].mean().sort_values(ascending=False).head(20)
Barchart(Moy_Points_Sum_by_WorkCountry, "", "", "")

MembersCount_by_WORK_CITY = pd.value_counts(csv["WORK_CITY"], sort=True).head(20)
Barchart(MembersCount_by_WORK_CITY, "", "", "")

MembersCount_by_WORK_PROVINCE = pd.value_counts(csv["WORK_PROVINCE"], sort=True).head(20)
Barchart(MembersCount_by_WORK_PROVINCE, "", "", "")

# Pair Plots : (Seaborn)



