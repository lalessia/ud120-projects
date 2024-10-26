#!/usr/bin/python3
import os
import joblib
import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(os.path.abspath("../tools/"))
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = joblib.load( open("../final_project/final_project_dataset.pkl", "rb") )
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below

salary = data[:, 0]
bonus = data[:, 1]



try:
   plt.scatter(salary, bonus, c ="blue")
except NameError:
    pass
plt.scatter(salary, bonus)
plt.show()


# Trova il massimo valore per bonus e salario
max_salary = max(salary)
max_bonus = max(bonus)

# Identifica la chiave che contiene i valori massimi
outlier_key = None
for key, value in data_dict.items():
    # Controlla che i valori non siano "NaN" per evitarne il confronto
    if value["salary"] == max_salary or value["bonus"] == max_bonus:
        outlier_key = key
        break

print("La chiave dell'outlier è:", outlier_key)

# Rimuovi l'outlier
data_dict.pop(outlier_key, 0)

# Rielabora i dati senza l'outlier
data = featureFormat(data_dict, features)

# Aggiorna le variabili salary e bonus
salary = data[:, 0]
bonus = data[:, 1]

# Visualizza il nuovo scatterplot
plt.scatter(salary, bonus, c="blue")
plt.xlabel("Salary")
plt.ylabel("Bonus")
plt.show()

# Lista per i nomi degli outliers
outliers = []

# Loop su ogni persona nel dizionario
for name, features in data_dict.items():
    salary = features.get("salary", 0)
    bonus = features.get("bonus", 0)
    
    # Controlla se i dati rispettano le condizioni richieste
    if salary != "NaN" and bonus != "NaN" and salary > 1_000_000 and bonus > 5_000_000:
        outliers.append((name, salary, bonus))

# Visualizza i risultati
for name, salary, bonus in outliers:
    print(f"Name: {name}, Salary: {salary}, Bonus: {bonus}")