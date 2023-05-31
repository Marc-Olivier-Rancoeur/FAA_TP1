import pandas as pd
import matplotlib.pyplot as plt

# I.1 _ Importation du fichier CSV #
datas = pd.read_csv("hourly_consumption_data.csv", dtype={"Datetime": "str", "PJME_MW": "float"}, parse_dates=["Datetime"], date_format="%Y-%m-%d %H:%M:%S")
print(datas.describe()) # affichage des informations principales du set
print(datas.head())
# Trie des données
datas.sort_values(by=["Datetime"], inplace=True, ignore_index=True)
####

# I.2 _ Vérification des données manquantes #
print("Vérification des valeurs manquantes")
if datas["Datetime"].count() == datas["PJME_MW"].count(): # vérification qu'on a bien la consommation pour chaque date
    oneHour = pd.Timedelta(hours=1) # création d'une heure afin de faire la vérification heure par heure
    valeurDate = datas["Datetime"].get(0)
    for i in range(1, datas["Datetime"].count()):
        if valeurDate + oneHour != datas["Datetime"].get(i): # vérification par rapport à l'heure précédente +1
            if valeurDate == datas["Datetime"].get(i): # test spécial doublon
                print("Doublon de date : " + str(valeurDate))
            else:
                print("Valeur manquante entre : " + str(valeurDate) + " et : " + str(datas["Datetime"].get(i)))
        valeurDate = datas["Datetime"].get(i)
else:
    print("Valeurs manquantes")
    exit(-1)
####

# I.3 _ Visualisation des données #
plt.figure(figsize=(12, 6))
plt.plot(datas["Datetime"][:167], datas["PJME_MW"][:167]) # On affiche les données sur une semaine
plt.xlabel('Date-Time')
plt.ylabel('Consommation(MW)')
plt.title("Consommation en fonction du temps")
plt.show()
####


# II.1
from statsmodels.tsa.stattools import adfuller, kpss
valeurs = datas["PJME_MW"]
# ADF
result = adfuller(valeurs)

print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t', key, ':', value)

# KPSS
result = kpss(valeurs, nlags=720)
print('KPSS Statistic:', result[0])
print('p-value:', result[1])
print('Lags Used:', result[2])
print('Critical Values:')
for key, value in result[3].items():
    print('\t', key, ':', value)
####

# II.2
# D'après les tests, la série est stationnaire.
####


# III.1
import seaborn as sns

# Création des variables pour la matrice de corrélation
datas["year"] = datas["Datetime"].dt.year
datas["month"] = datas["Datetime"].dt.month
datas["day"] = datas["Datetime"].dt.day
datas["hour"] = datas["Datetime"].dt.hour
datas["X-6"] = datas["PJME_MW"].shift(6)
datas["X-5"] = datas["PJME_MW"].shift(5)
datas["X-4"] = datas["PJME_MW"].shift(4)
datas["X-3"] = datas["PJME_MW"].shift(3)
datas["X-2"] = datas["PJME_MW"].shift(2)
datas["X-1"] = datas["PJME_MW"].shift(1)

matrice_correlation = datas.corr()

sns.heatmap(matrice_correlation, annot=True) # Transformation de la matrice en heatmap
plt.show()

datas = datas.drop(["year", "month", "day", "hour", "X-6", "X-5", "X-4", "X-3", "X-2", "X-1"], axis=1) # destruction des variables qui ne seront pas utilisées par la suite"""
####

# III.2
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1)) # définition du scaler entre -1 et 1.

datas["hour"] = datas["Datetime"].dt.hour
datas.drop("Datetime", axis=1, inplace=True) # destruction du datetime après avoir extrait l'heure
datas2 = pd.DataFrame()
datas2["Y"] = datas["PJME_MW"].copy() # prédéfinition du Y pour la suite qui sera séparé du PJME_MW

datas_ = scaler.fit_transform(datas) # fit de toutes les données entre -1 et 1.
datas = pd.DataFrame(datas_, columns=datas.columns)
datas["Y"] = scaler.fit_transform(datas2["Y"].values.reshape(-1, 1)) # redéfinition pour le Y
####
# III.2.a
# Création des dataframes de données de lag
datas_lag_6 = pd.DataFrame()
datas_lag_12 = pd.DataFrame()
datas_lag_24 = pd.DataFrame()

#remplissage des dataframes
datas_lag_6["hour"] = datas["hour"]
datas_lag_12["hour"] = datas["hour"]
datas_lag_24["hour"] = datas["hour"]

datas_lag_6["X"] = datas["PJME_MW"].shift(6)
datas_lag_12["X"] = datas["PJME_MW"].shift(12)
datas_lag_24["X"] = datas["PJME_MW"].shift(24)

datas_lag_6["Y"] = datas["Y"]
datas_lag_12["Y"] = datas["Y"]
datas_lag_24["Y"] = datas["Y"]
####
# III.2.b
# Création des dataframes de données de rolling
datas_rol_6 = pd.DataFrame()
datas_rol_12 = pd.DataFrame()
datas_rol_24 = pd.DataFrame()

# remplissage des dataframe
datas_rol_6["hour"] = datas["hour"]
datas_rol_12["hour"] = datas["hour"]
datas_rol_24["hour"] = datas["hour"]

datas_rol_6["X_mean"] = datas["PJME_MW"].rolling(window=6).mean().shift(6) # shift nécessaire en plus du glissement pour bien prédire dans 6h
datas_rol_12["X_mean"] = datas["PJME_MW"].rolling(window=12).mean().shift(12)
datas_rol_24["X_mean"] = datas["PJME_MW"].rolling(window=24).mean().shift(24)

datas_rol_6["X_min"] = datas["PJME_MW"].rolling(window=6).min().shift(6)
datas_rol_12["X_min"] = datas["PJME_MW"].rolling(window=12).min().shift(12)
datas_rol_24["X_min"] = datas["PJME_MW"].rolling(window=24).min().shift(24)

datas_rol_6["X_max"] = datas["PJME_MW"].rolling(window=6).max().shift(6)
datas_rol_12["X_max"] = datas["PJME_MW"].rolling(window=12).max().shift(12)
datas_rol_24["X_max"] = datas["PJME_MW"].rolling(window=24).max().shift(24)

datas_rol_6["X_std"] = datas["PJME_MW"].rolling(window=6).std().shift(6)
datas_rol_12["X_std"] = datas["PJME_MW"].rolling(window=12).std().shift(12)
datas_rol_24["X_std"] = datas["PJME_MW"].rolling(window=24).std().shift(24)

datas_rol_6["Y"] = datas["PJME_MW"]
datas_rol_12["Y"] = datas["PJME_MW"]
datas_rol_24["Y"] = datas["PJME_MW"]
####

# III.3

# - Mise en forme des données -

import numpy as np

# définition de la taille des train et test sets
train_ratio = 0.8
train_size = int(len(datas) * train_ratio)

# Création des X et Y
X_lag_6 = datas_lag_6.drop("Y", axis=1)
Y_lag_6 = datas_lag_6["Y"]

# Division en train et test sets
X_lag_6_train = X_lag_6[6:train_size]
X_lag_6_test = X_lag_6[train_size:]
Y_lag_6_train = Y_lag_6[6:train_size]
Y_lag_6_test = Y_lag_6[train_size:]

# Reshaping des données afin qu'elle rentrent dans nos modèles
X_lag_6_train = np.reshape(X_lag_6_train, (X_lag_6_train.shape[0], X_lag_6_train.shape[1], 1))
X_lag_6_test = np.reshape(X_lag_6_test, (X_lag_6_test.shape[0], X_lag_6_test.shape[1], 1))

# On recommence pour chaque ...
X_lag_12 = datas_lag_12.drop("Y", axis=1)
Y_lag_12 = datas_lag_12["Y"]

X_lag_12_train = X_lag_12[12:train_size]
X_lag_12_test = X_lag_12[train_size:]
Y_lag_12_train = Y_lag_12[12:train_size]
Y_lag_12_test = Y_lag_12[train_size:]

X_lag_12_train = np.reshape(X_lag_12_train, (X_lag_12_train.shape[0], X_lag_12_train.shape[1], 1))
X_lag_12_test = np.reshape(X_lag_12_test, (X_lag_12_test.shape[0], X_lag_12_test.shape[1], 1))

X_lag_24 = datas_lag_24.drop("Y", axis=1)
Y_lag_24 = datas_lag_24["Y"]

X_lag_24_train = X_lag_24[24:train_size]
X_lag_24_test = X_lag_24[train_size:]
Y_lag_24_train = Y_lag_24[24:train_size]
Y_lag_24_test = Y_lag_24[train_size:]

X_lag_24_train = np.reshape(X_lag_24_train, (X_lag_24_train.shape[0], X_lag_24_train.shape[1], 1))
X_lag_24_test = np.reshape(X_lag_24_test, (X_lag_24_test.shape[0], X_lag_24_test.shape[1], 1))

# Pareil pour le rolling
X_rol_6_mean = datas_rol_6.drop(["X_min", "X_max", "X_std", "Y"], axis=1)
X_rol_6_min = datas_rol_6.drop(["X_mean", "X_max", "X_std", "Y"], axis=1)
X_rol_6_max = datas_rol_6.drop(["X_mean", "X_min", "X_std", "Y"], axis=1)
X_rol_6_std = datas_rol_6.drop(["X_mean", "X_min", "X_max", "Y"], axis=1)
Y_rol_6 = datas_rol_6["Y"]

X_rol_6_mean_train = X_rol_6_mean[12:train_size]
X_rol_6_mean_test = X_rol_6_mean[train_size:]
X_rol_6_min_train = X_rol_6_min[12:train_size]
X_rol_6_min_test = X_rol_6_min[train_size:]
X_rol_6_max_train = X_rol_6_max[12:train_size]
X_rol_6_max_test = X_rol_6_max[train_size:]
X_rol_6_std_train = X_rol_6_std[12:train_size]
X_rol_6_std_test = X_rol_6_std[train_size:]
Y_rol_6_train = Y_rol_6[12:train_size]
Y_rol_6_test = Y_rol_6[train_size:]

X_rol_6_mean_train = np.reshape(X_rol_6_mean_train, (X_rol_6_mean_train.shape[0], X_rol_6_mean_train.shape[1], 1))
X_rol_6_mean_test = np.reshape(X_rol_6_mean_test, (X_rol_6_mean_test.shape[0], X_rol_6_mean_test.shape[1], 1))
X_rol_6_min_train = np.reshape(X_rol_6_min_train, (X_rol_6_min_train.shape[0], X_rol_6_min_train.shape[1], 1))
X_rol_6_min_test = np.reshape(X_rol_6_min_test, (X_rol_6_min_test.shape[0], X_rol_6_min_test.shape[1], 1))
X_rol_6_max_train = np.reshape(X_rol_6_max_train, (X_rol_6_max_train.shape[0], X_rol_6_max_train.shape[1], 1))
X_rol_6_max_test = np.reshape(X_rol_6_max_test, (X_rol_6_max_test.shape[0], X_rol_6_max_test.shape[1], 1))
X_rol_6_std_train = np.reshape(X_rol_6_std_train, (X_rol_6_std_train.shape[0], X_rol_6_std_train.shape[1], 1))
X_rol_6_std_test = np.reshape(X_rol_6_std_test, (X_rol_6_std_test.shape[0], X_rol_6_std_test.shape[1], 1))


X_rol_12_mean = datas_rol_12.drop(["X_min", "X_max", "X_std", "Y"], axis=1)
X_rol_12_min = datas_rol_12.drop(["X_mean", "X_max", "X_std", "Y"], axis=1)
X_rol_12_max = datas_rol_12.drop(["X_mean", "X_min", "X_std", "Y"], axis=1)
X_rol_12_std = datas_rol_12.drop(["X_mean", "X_min", "X_max", "Y"], axis=1)
Y_rol_12 = datas_rol_12["Y"]

X_rol_12_mean_train = X_rol_12_mean[24:train_size]
X_rol_12_mean_test = X_rol_12_mean[train_size:]
X_rol_12_min_train = X_rol_12_min[24:train_size]
X_rol_12_min_test = X_rol_12_min[train_size:]
X_rol_12_max_train = X_rol_12_max[24:train_size]
X_rol_12_max_test = X_rol_12_max[train_size:]
X_rol_12_std_train = X_rol_12_std[24:train_size]
X_rol_12_std_test = X_rol_12_std[train_size:]
Y_rol_12_train = Y_rol_12[24:train_size]
Y_rol_12_test = Y_rol_12[train_size:]

X_rol_12_mean_train = np.reshape(X_rol_12_mean_train, (X_rol_12_mean_train.shape[0], X_rol_12_mean_train.shape[1], 1))
X_rol_12_mean_test = np.reshape(X_rol_12_mean_test, (X_rol_12_mean_test.shape[0], X_rol_12_mean_test.shape[1], 1))
X_rol_12_min_train = np.reshape(X_rol_12_min_train, (X_rol_12_min_train.shape[0], X_rol_12_min_train.shape[1], 1))
X_rol_12_min_test = np.reshape(X_rol_12_min_test, (X_rol_12_min_test.shape[0], X_rol_12_min_test.shape[1], 1))
X_rol_12_max_train = np.reshape(X_rol_12_max_train, (X_rol_12_max_train.shape[0], X_rol_12_max_train.shape[1], 1))
X_rol_12_max_test = np.reshape(X_rol_12_max_test, (X_rol_12_max_test.shape[0], X_rol_12_max_test.shape[1], 1))
X_rol_12_std_train = np.reshape(X_rol_12_std_train, (X_rol_12_std_train.shape[0], X_rol_12_std_train.shape[1], 1))
X_rol_12_std_test = np.reshape(X_rol_12_std_test, (X_rol_12_std_test.shape[0], X_rol_12_std_test.shape[1], 1))


X_rol_24_mean = datas_rol_24.drop(["X_min", "X_max", "X_std", "Y"], axis=1)
X_rol_24_min = datas_rol_24.drop(["X_mean", "X_max", "X_std", "Y"], axis=1)
X_rol_24_max = datas_rol_24.drop(["X_mean", "X_min", "X_std", "Y"], axis=1)
X_rol_24_std = datas_rol_24.drop(["X_mean", "X_min", "X_max", "Y"], axis=1)
Y_rol_24 = datas_rol_24["Y"]

X_rol_24_mean_train = X_rol_24_mean[48:train_size]
X_rol_24_mean_test = X_rol_24_mean[train_size:]
X_rol_24_min_train = X_rol_24_min[48:train_size]
X_rol_24_min_test = X_rol_24_min[train_size:]
X_rol_24_max_train = X_rol_24_max[48:train_size]
X_rol_24_max_test = X_rol_24_max[train_size:]
X_rol_24_std_train = X_rol_24_std[48:train_size]
X_rol_24_std_test = X_rol_24_std[train_size:]
Y_rol_24_train = Y_rol_24[48:train_size]
Y_rol_24_test = Y_rol_24[train_size:]

X_rol_24_mean_train = np.reshape(X_rol_24_mean_train, (X_rol_24_mean_train.shape[0], X_rol_24_mean_train.shape[1], 1))
X_rol_24_mean_test = np.reshape(X_rol_24_mean_test, (X_rol_24_mean_test.shape[0], X_rol_24_mean_test.shape[1], 1))
X_rol_24_min_train = np.reshape(X_rol_24_min_train, (X_rol_24_min_train.shape[0], X_rol_24_min_train.shape[1], 1))
X_rol_24_min_test = np.reshape(X_rol_24_min_test, (X_rol_24_min_test.shape[0], X_rol_24_min_test.shape[1], 1))
X_rol_24_max_train = np.reshape(X_rol_24_max_train, (X_rol_24_max_train.shape[0], X_rol_24_max_train.shape[1], 1))
X_rol_24_max_test = np.reshape(X_rol_24_max_test, (X_rol_24_max_test.shape[0], X_rol_24_max_test.shape[1], 1))
X_rol_24_std_train = np.reshape(X_rol_24_std_train, (X_rol_24_std_train.shape[0], X_rol_24_std_train.shape[1], 1))
X_rol_24_std_test = np.reshape(X_rol_24_std_test, (X_rol_24_std_test.shape[0], X_rol_24_std_test.shape[1], 1))

# - Début de l'apprentissage des modèles -

from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print(" - LSTM lag 6 - ")
model_lag_6 = Sequential([LSTM(50, activation='relu'), Dense(1)]) # définition du modèle LSTM
model_lag_6.compile(optimizer='adam', loss='mse')
model_lag_6.fit(X_lag_6_train, Y_lag_6_train, epochs=10, batch_size=168, verbose=0) # entrainement du modèle

Y_lag_6_pred = model_lag_6.predict(X_lag_6_test) # prédiction du model avec le test set
Y_lag_6_pred_sc = scaler.inverse_transform(Y_lag_6_pred) # rescaling de la prediction
Y_lag_6_test_sc = scaler.inverse_transform(Y_lag_6_test.values.reshape(-1, 1)) # rescaling du Y de test

# Calcul des différentes variables de qualité
mae_lag_6 = mean_absolute_error(Y_lag_6_test_sc, Y_lag_6_pred_sc)
rmse_lag_6 = np.sqrt(mean_squared_error(Y_lag_6_test_sc, Y_lag_6_pred_sc))
r2_lag_6 = r2_score(Y_lag_6_test_sc, Y_lag_6_pred_sc)
mape_lag_6 = np.mean(np.abs((Y_lag_6_test_sc - Y_lag_6_pred_sc) / Y_lag_6_test_sc)) * 100

# Affichage de celles-ci
print("MAE:", mae_lag_6)
print("RMSE:", rmse_lag_6)
print("R2:", r2_lag_6)
print("MAPE:", mape_lag_6)

# Création d'un graphique pour observer un résultat dans le test
plt.figure(figsize=(12, 6))
plt.plot(Y_lag_6_test_sc, label='Vraies valeurs')
plt.plot(Y_lag_6_pred_sc, label='Prédictions')
plt.xlabel('Indice')
plt.ylabel('Consommation')
plt.title('LSTM lag 6')
plt.legend()
plt.xlim([2000, 2167]) # récupération d'est données sur 1 semaine dans le test set
plt.show()

# Répétition de la même chose
print(" - LSTM lag 12 - ")
model_lag_12 = Sequential([LSTM(50, activation='relu'), Dense(1)])
model_lag_12.compile(optimizer='adam', loss='mse')
model_lag_12.fit(X_lag_12_train, Y_lag_12_train, epochs=10, batch_size=168, verbose=0)

Y_lag_12_pred = model_lag_12.predict(X_lag_12_test)
Y_lag_12_pred_sc = scaler.inverse_transform(Y_lag_12_pred)
Y_lag_12_test_sc = scaler.inverse_transform(Y_lag_12_test.values.reshape(-1, 1))

mae_lag_12 = mean_absolute_error(Y_lag_12_test_sc, Y_lag_12_pred_sc)
rmse_lag_12 = np.sqrt(mean_squared_error(Y_lag_12_test_sc, Y_lag_12_pred_sc))
r2_lag_12 = r2_score(Y_lag_12_test_sc, Y_lag_12_pred_sc)
mape_lag_12 = np.mean(np.abs((Y_lag_12_test_sc - Y_lag_12_pred_sc) / Y_lag_12_test_sc)) * 100

print("MAE:", mae_lag_12)
print("RMSE:", rmse_lag_12)
print("R2:", r2_lag_12)
print("MAPE:", mape_lag_12)

plt.figure(figsize=(12, 6))
plt.plot(Y_lag_12_test_sc, label='Vraies valeurs')
plt.plot(Y_lag_12_pred_sc, label='Prédictions')
plt.xlabel('Indice')
plt.ylabel('Consommation')
plt.title('LSTM lag 12')
plt.legend()
plt.xlim([2000, 2167])
plt.show()


print(" - LSTM lag 24 - ")
model_lag_24 = Sequential([LSTM(50, activation='relu'), Dense(1)])
model_lag_24.compile(optimizer='adam', loss='mse')
model_lag_24.fit(X_lag_24_train, Y_lag_24_train, epochs=10, batch_size=168, verbose=0)

Y_lag_24_pred = model_lag_24.predict(X_lag_24_test)
Y_lag_24_pred_sc = scaler.inverse_transform(Y_lag_24_pred)
Y_lag_24_test_sc = scaler.inverse_transform(Y_lag_24_test.values.reshape(-1, 1))

mae_lag_24 = mean_absolute_error(Y_lag_24_test_sc, Y_lag_24_pred_sc)
rmse_lag_24 = np.sqrt(mean_squared_error(Y_lag_24_test_sc, Y_lag_24_pred_sc))
r2_lag_24 = r2_score(Y_lag_24_test_sc, Y_lag_24_pred_sc)
mape_lag_24 = np.mean(np.abs((Y_lag_24_test_sc - Y_lag_24_pred_sc) / Y_lag_24_test_sc)) * 100

print("MAE:", mae_lag_24)
print("RMSE:", rmse_lag_24)
print("R2:", r2_lag_24)
print("MAPE:", mape_lag_24)

plt.figure(figsize=(12, 6))
plt.plot(Y_lag_24_test_sc, label='Vraies valeurs')
plt.plot(Y_lag_24_pred_sc, label='Prédictions')
plt.xlabel('Indice')
plt.ylabel('Consommation')
plt.title('LSTM lag 24')
plt.legend()
plt.xlim([2000, 2167])
plt.show()

# Pareil avec le rol
print(" - LSTM rol 6 mean - ")
model_rol_6_mean = Sequential([LSTM(50, activation='relu'), Dense(1)])
model_rol_6_mean.compile(optimizer='adam', loss='mse')
model_rol_6_mean.fit(X_rol_6_mean_train, Y_rol_6_train, epochs=10, batch_size=168, verbose=0)

Y_rol_6_mean_pred = model_rol_6_mean.predict(X_rol_6_mean_test)
Y_rol_6_mean_pred_sc = scaler.inverse_transform(Y_rol_6_mean_pred)
Y_rol_6_mean_test_sc = scaler.inverse_transform(Y_rol_6_test.values.reshape(-1, 1))

mae_rol_6_mean = mean_absolute_error(Y_rol_6_mean_test_sc, Y_rol_6_mean_pred_sc)
rmse_rol_6_mean = np.sqrt(mean_squared_error(Y_rol_6_mean_test_sc, Y_rol_6_mean_pred_sc))
r2_rol_6_mean = r2_score(Y_rol_6_mean_test_sc, Y_rol_6_mean_pred_sc)
mape_rol_6_mean = np.mean(np.abs((Y_rol_6_mean_test_sc - Y_rol_6_mean_pred_sc) / Y_rol_6_mean_test_sc)) * 100

print("MAE:", mae_rol_6_mean)
print("RMSE:", rmse_rol_6_mean)
print("R2:", r2_rol_6_mean)
print("MAPE:", mape_rol_6_mean)

plt.figure(figsize=(12, 6))
plt.plot(Y_rol_6_mean_test_sc, label='Vraies valeurs')
plt.plot(Y_rol_6_mean_pred_sc, label='Prédictions')
plt.xlabel('Indice')
plt.ylabel('Consommation')
plt.title('LSTM rol 6 mean')
plt.legend()
plt.xlim([2000, 2167])
plt.show()


print(" - LSTM rol 12 mean - ")
model_rol_12_mean = Sequential([LSTM(50, activation='relu'), Dense(1)])
model_rol_12_mean.compile(optimizer='adam', loss='mse')
model_rol_12_mean.fit(X_rol_12_mean_train, Y_rol_12_train, epochs=10, batch_size=168, verbose=0)

Y_rol_12_mean_pred = model_rol_12_mean.predict(X_rol_12_mean_test)
Y_rol_12_mean_pred_sc = scaler.inverse_transform(Y_rol_12_mean_pred)
Y_rol_12_mean_test_sc = scaler.inverse_transform(Y_rol_12_test.values.reshape(-1, 1))

mae_rol_12_mean = mean_absolute_error(Y_rol_12_mean_test_sc, Y_rol_12_mean_pred_sc)
rmse_rol_12_mean = np.sqrt(mean_squared_error(Y_rol_12_mean_test_sc, Y_rol_12_mean_pred_sc))
r2_rol_12_mean = r2_score(Y_rol_12_mean_test_sc, Y_rol_12_mean_pred_sc)
mape_rol_12_mean = np.mean(np.abs((Y_rol_12_mean_test_sc - Y_rol_12_mean_pred_sc) / Y_rol_12_mean_test_sc)) * 100

print("MAE:", mae_rol_12_mean)
print("RMSE:", rmse_rol_12_mean)
print("R2:", r2_rol_12_mean)
print("MAPE:", mape_rol_12_mean)

plt.figure(figsize=(12, 6))
plt.plot(Y_rol_12_mean_test_sc, label='Vraies valeurs')
plt.plot(Y_rol_12_mean_pred_sc, label='Prédictions')
plt.xlabel('Indice')
plt.ylabel('Consommation')
plt.title('LSTM rol 12 mean')
plt.legend()
plt.xlim([2000, 2167])
plt.show()


print(" - LSTM rol 24 mean - ")
model_rol_24_mean = Sequential([LSTM(50, activation='relu'), Dense(1)])
model_rol_24_mean.compile(optimizer='adam', loss='mse')
model_rol_24_mean.fit(X_rol_24_mean_train, Y_rol_24_train, epochs=10, batch_size=168, verbose=0)

Y_rol_24_mean_pred = model_rol_24_mean.predict(X_rol_24_mean_test)
Y_rol_24_mean_pred_sc = scaler.inverse_transform(Y_rol_24_mean_pred)
Y_rol_24_mean_test_sc = scaler.inverse_transform(Y_rol_24_test.values.reshape(-1, 1))

mae_rol_24_mean = mean_absolute_error(Y_rol_24_mean_test_sc, Y_rol_24_mean_pred_sc)
rmse_rol_24_mean = np.sqrt(mean_squared_error(Y_rol_24_mean_test_sc, Y_rol_24_mean_pred_sc))
r2_rol_24_mean = r2_score(Y_rol_24_mean_test_sc, Y_rol_24_mean_pred_sc)
mape_rol_24_mean = np.mean(np.abs((Y_rol_24_mean_test_sc - Y_rol_24_mean_pred_sc) / Y_rol_24_mean_test_sc)) * 100

print("MAE:", mae_rol_24_mean)
print("RMSE:", rmse_rol_24_mean)
print("R2:", r2_rol_24_mean)
print("MAPE:", mape_rol_24_mean)

plt.figure(figsize=(12, 6))
plt.plot(Y_rol_24_mean_test_sc, label='Vraies valeurs')
plt.plot(Y_rol_24_mean_pred_sc, label='Prédictions')
plt.xlabel('Indice')
plt.ylabel('Consommation')
plt.title('LSTM rol 24 mean')
plt.legend()
plt.xlim([2000, 2167])
plt.show()



print(" - LSTM rol 6 min - ")
model_rol_6_min = Sequential([LSTM(50, activation='relu'), Dense(1)])
model_rol_6_min.compile(optimizer='adam', loss='mse')
model_rol_6_min.fit(X_rol_6_min_train, Y_rol_6_train, epochs=10, batch_size=168, verbose=0)

Y_rol_6_min_pred = model_rol_6_min.predict(X_rol_6_min_test)
Y_rol_6_min_pred_sc = scaler.inverse_transform(Y_rol_6_min_pred)
Y_rol_6_min_test_sc = scaler.inverse_transform(Y_rol_6_test.values.reshape(-1, 1))

mae_rol_6_min = mean_absolute_error(Y_rol_6_min_test_sc, Y_rol_6_min_pred_sc)
rmse_rol_6_min = np.sqrt(mean_squared_error(Y_rol_6_min_test_sc, Y_rol_6_min_pred_sc))
r2_rol_6_min = r2_score(Y_rol_6_min_test_sc, Y_rol_6_min_pred_sc)
mape_rol_6_min = np.min(np.abs((Y_rol_6_min_test_sc - Y_rol_6_min_pred_sc) / Y_rol_6_min_test_sc)) * 100

print("MAE:", mae_rol_6_min)
print("RMSE:", rmse_rol_6_min)
print("R2:", r2_rol_6_min)
print("MAPE:", mape_rol_6_min)

plt.figure(figsize=(12, 6))
plt.plot(Y_rol_6_min_test_sc, label='Vraies valeurs')
plt.plot(Y_rol_6_min_pred_sc, label='Prédictions')
plt.xlabel('Indice')
plt.ylabel('Consommation')
plt.title('LSTM rol 6 min')
plt.legend()
plt.xlim([2000, 2167])
plt.show()


print(" - LSTM rol 12 min - ")
model_rol_12_min = Sequential([LSTM(50, activation='relu'), Dense(1)])
model_rol_12_min.compile(optimizer='adam', loss='mse')
model_rol_12_min.fit(X_rol_12_min_train, Y_rol_12_train, epochs=10, batch_size=168, verbose=0)

Y_rol_12_min_pred = model_rol_12_min.predict(X_rol_12_min_test)
Y_rol_12_min_pred_sc = scaler.inverse_transform(Y_rol_12_min_pred)
Y_rol_12_min_test_sc = scaler.inverse_transform(Y_rol_12_test.values.reshape(-1, 1))

mae_rol_12_min = mean_absolute_error(Y_rol_12_min_test_sc, Y_rol_12_min_pred_sc)
rmse_rol_12_min = np.sqrt(mean_squared_error(Y_rol_12_min_test_sc, Y_rol_12_min_pred_sc))
r2_rol_12_min = r2_score(Y_rol_12_min_test_sc, Y_rol_12_min_pred_sc)
mape_rol_12_min = np.min(np.abs((Y_rol_12_min_test_sc - Y_rol_12_min_pred_sc) / Y_rol_12_min_test_sc)) * 100

print("MAE:", mae_rol_12_min)
print("RMSE:", rmse_rol_12_min)
print("R2:", r2_rol_12_min)
print("MAPE:", mape_rol_12_min)

plt.figure(figsize=(12, 6))
plt.plot(Y_rol_12_min_test_sc, label='Vraies valeurs')
plt.plot(Y_rol_12_min_pred_sc, label='Prédictions')
plt.xlabel('Indice')
plt.ylabel('Consommation')
plt.title('LSTM rol 12 min')
plt.legend()
plt.xlim([2000, 2167])
plt.show()


print(" - LSTM rol 24 min - ")
model_rol_24_min = Sequential([LSTM(50, activation='relu'), Dense(1)])
model_rol_24_min.compile(optimizer='adam', loss='mse')
model_rol_24_min.fit(X_rol_24_min_train, Y_rol_24_train, epochs=10, batch_size=168, verbose=0)

Y_rol_24_min_pred = model_rol_24_min.predict(X_rol_24_min_test)
Y_rol_24_min_pred_sc = scaler.inverse_transform(Y_rol_24_min_pred)
Y_rol_24_min_test_sc = scaler.inverse_transform(Y_rol_24_test.values.reshape(-1, 1))

mae_rol_24_min = mean_absolute_error(Y_rol_24_min_test_sc, Y_rol_24_min_pred_sc)
rmse_rol_24_min = np.sqrt(mean_squared_error(Y_rol_24_min_test_sc, Y_rol_24_min_pred_sc))
r2_rol_24_min = r2_score(Y_rol_24_min_test_sc, Y_rol_24_min_pred_sc)
mape_rol_24_min = np.min(np.abs((Y_rol_24_min_test_sc - Y_rol_24_min_pred_sc) / Y_rol_24_min_test_sc)) * 100

print("MAE:", mae_rol_24_min)
print("RMSE:", rmse_rol_24_min)
print("R2:", r2_rol_24_min)
print("MAPE:", mape_rol_24_min)

plt.figure(figsize=(12, 6))
plt.plot(Y_rol_24_min_test_sc, label='Vraies valeurs')
plt.plot(Y_rol_24_min_pred_sc, label='Prédictions')
plt.xlabel('Indice')
plt.ylabel('Consommation')
plt.title('LSTM rol 24 min')
plt.legend()
plt.xlim([2000, 2167])
plt.show()



print(" - LSTM rol 6 max - ")
model_rol_6_max = Sequential([LSTM(50, activation='relu'), Dense(1)])
model_rol_6_max.compile(optimizer='adam', loss='mse')
model_rol_6_max.fit(X_rol_6_max_train, Y_rol_6_train, epochs=10, batch_size=168, verbose=0)

Y_rol_6_max_pred = model_rol_6_max.predict(X_rol_6_max_test)
Y_rol_6_max_pred_sc = scaler.inverse_transform(Y_rol_6_max_pred)
Y_rol_6_max_test_sc = scaler.inverse_transform(Y_rol_6_test.values.reshape(-1, 1))

mae_rol_6_max = mean_absolute_error(Y_rol_6_max_test_sc, Y_rol_6_max_pred_sc)
rmse_rol_6_max = np.sqrt(mean_squared_error(Y_rol_6_max_test_sc, Y_rol_6_max_pred_sc))
r2_rol_6_max = r2_score(Y_rol_6_max_test_sc, Y_rol_6_max_pred_sc)
mape_rol_6_max = np.max(np.abs((Y_rol_6_max_test_sc - Y_rol_6_max_pred_sc) / Y_rol_6_max_test_sc)) * 100

print("MAE:", mae_rol_6_max)
print("RMSE:", rmse_rol_6_max)
print("R2:", r2_rol_6_max)
print("MAPE:", mape_rol_6_max)

plt.figure(figsize=(12, 6))
plt.plot(Y_rol_6_max_test_sc, label='Vraies valeurs')
plt.plot(Y_rol_6_max_pred_sc, label='Prédictions')
plt.xlabel('Indice')
plt.ylabel('Consommation')
plt.title('LSTM rol 6 max')
plt.legend()
plt.xlim([2000, 2167])
plt.show()


print(" - LSTM rol 12 max - ")
model_rol_12_max = Sequential([LSTM(50, activation='relu'), Dense(1)])
model_rol_12_max.compile(optimizer='adam', loss='mse')
model_rol_12_max.fit(X_rol_12_max_train, Y_rol_12_train, epochs=10, batch_size=168, verbose=0)

Y_rol_12_max_pred = model_rol_12_max.predict(X_rol_12_max_test)
Y_rol_12_max_pred_sc = scaler.inverse_transform(Y_rol_12_max_pred)
Y_rol_12_max_test_sc = scaler.inverse_transform(Y_rol_12_test.values.reshape(-1, 1))

mae_rol_12_max = mean_absolute_error(Y_rol_12_max_test_sc, Y_rol_12_max_pred_sc)
rmse_rol_12_max = np.sqrt(mean_squared_error(Y_rol_12_max_test_sc, Y_rol_12_max_pred_sc))
r2_rol_12_max = r2_score(Y_rol_12_max_test_sc, Y_rol_12_max_pred_sc)
mape_rol_12_max = np.max(np.abs((Y_rol_12_max_test_sc - Y_rol_12_max_pred_sc) / Y_rol_12_max_test_sc)) * 100

print("MAE:", mae_rol_12_max)
print("RMSE:", rmse_rol_12_max)
print("R2:", r2_rol_12_max)
print("MAPE:", mape_rol_12_max)

plt.figure(figsize=(12, 6))
plt.plot(Y_rol_12_max_test_sc, label='Vraies valeurs')
plt.plot(Y_rol_12_max_pred_sc, label='Prédictions')
plt.xlabel('Indice')
plt.ylabel('Consommation')
plt.title('LSTM rol 12 max')
plt.legend()
plt.xlim([2000, 2167])
plt.show()


print(" - LSTM rol 24 max - ")
model_rol_24_max = Sequential([LSTM(50, activation='relu'), Dense(1)])
model_rol_24_max.compile(optimizer='adam', loss='mse')
model_rol_24_max.fit(X_rol_24_max_train, Y_rol_24_train, epochs=10, batch_size=168, verbose=0)

Y_rol_24_max_pred = model_rol_24_max.predict(X_rol_24_max_test)
Y_rol_24_max_pred_sc = scaler.inverse_transform(Y_rol_24_max_pred)
Y_rol_24_max_test_sc = scaler.inverse_transform(Y_rol_24_test.values.reshape(-1, 1))

mae_rol_24_max = mean_absolute_error(Y_rol_24_max_test_sc, Y_rol_24_max_pred_sc)
rmse_rol_24_max = np.sqrt(mean_squared_error(Y_rol_24_max_test_sc, Y_rol_24_max_pred_sc))
r2_rol_24_max = r2_score(Y_rol_24_max_test_sc, Y_rol_24_max_pred_sc)
mape_rol_24_max = np.max(np.abs((Y_rol_24_max_test_sc - Y_rol_24_max_pred_sc) / Y_rol_24_max_test_sc)) * 100

print("MAE:", mae_rol_24_max)
print("RMSE:", rmse_rol_24_max)
print("R2:", r2_rol_24_max)
print("MAPE:", mape_rol_24_max)

plt.figure(figsize=(12, 6))
plt.plot(Y_rol_24_max_test_sc, label='Vraies valeurs')
plt.plot(Y_rol_24_max_pred_sc, label='Prédictions')
plt.xlabel('Indice')
plt.ylabel('Consommation')
plt.title('LSTM rol 24 max')
plt.legend()
plt.xlim([2000, 2167])
plt.show()



print(" - LSTM rol 6 std - ")
model_rol_6_std = Sequential([LSTM(50, activation='relu'), Dense(1)])
model_rol_6_std.compile(optimizer='adam', loss='mse')
model_rol_6_std.fit(X_rol_6_std_train, Y_rol_6_train, epochs=10, batch_size=168, verbose=0)

Y_rol_6_std_pred = model_rol_6_std.predict(X_rol_6_std_test)
Y_rol_6_std_pred_sc = scaler.inverse_transform(Y_rol_6_std_pred)
Y_rol_6_std_test_sc = scaler.inverse_transform(Y_rol_6_test.values.reshape(-1, 1))

mae_rol_6_std = mean_absolute_error(Y_rol_6_std_test_sc, Y_rol_6_std_pred_sc)
rmse_rol_6_std = np.sqrt(mean_squared_error(Y_rol_6_std_test_sc, Y_rol_6_std_pred_sc))
r2_rol_6_std = r2_score(Y_rol_6_std_test_sc, Y_rol_6_std_pred_sc)
mape_rol_6_std = np.std(np.abs((Y_rol_6_std_test_sc - Y_rol_6_std_pred_sc) / Y_rol_6_std_test_sc)) * 100

print("MAE:", mae_rol_6_std)
print("RMSE:", rmse_rol_6_std)
print("R2:", r2_rol_6_std)
print("MAPE:", mape_rol_6_std)

plt.figure(figsize=(12, 6))
plt.plot(Y_rol_6_std_test_sc, label='Vraies valeurs')
plt.plot(Y_rol_6_std_pred_sc, label='Prédictions')
plt.xlabel('Indice')
plt.ylabel('Consommation')
plt.title('LSTM rol 6 std')
plt.legend()
plt.xlim([2000, 2167])
plt.show()


print(" - LSTM rol 12 std - ")
model_rol_12_std = Sequential([LSTM(50, activation='relu'), Dense(1)])
model_rol_12_std.compile(optimizer='adam', loss='mse')
model_rol_12_std.fit(X_rol_12_std_train, Y_rol_12_train, epochs=10, batch_size=168, verbose=0)

Y_rol_12_std_pred = model_rol_12_std.predict(X_rol_12_std_test)
Y_rol_12_std_pred_sc = scaler.inverse_transform(Y_rol_12_std_pred)
Y_rol_12_std_test_sc = scaler.inverse_transform(Y_rol_12_test.values.reshape(-1, 1))

mae_rol_12_std = mean_absolute_error(Y_rol_12_std_test_sc, Y_rol_12_std_pred_sc)
rmse_rol_12_std = np.sqrt(mean_squared_error(Y_rol_12_std_test_sc, Y_rol_12_std_pred_sc))
r2_rol_12_std = r2_score(Y_rol_12_std_test_sc, Y_rol_12_std_pred_sc)
mape_rol_12_std = np.std(np.abs((Y_rol_12_std_test_sc - Y_rol_12_std_pred_sc) / Y_rol_12_std_test_sc)) * 100

print("MAE:", mae_rol_12_std)
print("RMSE:", rmse_rol_12_std)
print("R2:", r2_rol_12_std)
print("MAPE:", mape_rol_12_std)

plt.figure(figsize=(12, 6))
plt.plot(Y_rol_12_std_test_sc, label='Vraies valeurs')
plt.plot(Y_rol_12_std_pred_sc, label='Prédictions')
plt.xlabel('Indice')
plt.ylabel('Consommation')
plt.title('LSTM rol 12 std')
plt.legend()
plt.xlim([2000, 2167])
plt.show()


print(" - LSTM rol 24 std - ")
model_rol_24_std = Sequential([LSTM(50, activation='relu'), Dense(1)])
model_rol_24_std.compile(optimizer='adam', loss='mse')
model_rol_24_std.fit(X_rol_24_std_train, Y_rol_24_train, epochs=10, batch_size=168, verbose=0)

Y_rol_24_std_pred = model_rol_24_std.predict(X_rol_24_std_test)
Y_rol_24_std_pred_sc = scaler.inverse_transform(Y_rol_24_std_pred)
Y_rol_24_std_test_sc = scaler.inverse_transform(Y_rol_24_test.values.reshape(-1, 1))

mae_rol_24_std = mean_absolute_error(Y_rol_24_std_test_sc, Y_rol_24_std_pred_sc)
rmse_rol_24_std = np.sqrt(mean_squared_error(Y_rol_24_std_test_sc, Y_rol_24_std_pred_sc))
r2_rol_24_std = r2_score(Y_rol_24_std_test_sc, Y_rol_24_std_pred_sc)
mape_rol_24_std = np.std(np.abs((Y_rol_24_std_test_sc - Y_rol_24_std_pred_sc) / Y_rol_24_std_test_sc)) * 100

print("MAE:", mae_rol_24_std)
print("RMSE:", rmse_rol_24_std)
print("R2:", r2_rol_24_std)
print("MAPE:", mape_rol_24_std)

plt.figure(figsize=(12, 6))
plt.plot(Y_rol_24_std_test_sc, label='Vraies valeurs')
plt.plot(Y_rol_24_std_pred_sc, label='Prédictions')
plt.xlabel('Indice')
plt.ylabel('Consommation')
plt.title('LSTM rol 24 std')
plt.legend()
plt.xlim([2000, 2167])
plt.show()


# On passe a XGBoost
import xgboost as xgb

print(" - XGBoost lag 6 - ")
# Reformatage de nos données pour XGBoost
lag_6_train_m = xgb.DMatrix(X_lag_6_train.reshape(-1, X_lag_6_train.shape[1]), label=Y_lag_6_train, enable_categorical=True)
lag_6_test_m = xgb.DMatrix(X_lag_6_test.reshape(-1, X_lag_6_test.shape[1]), label=Y_lag_6_test, enable_categorical=True)

model_lag_6_xgb = xgb.train(params={"objective":"reg:squarederror"}, dtrain=lag_6_train_m, num_boost_round=100) # entrainement du modèle

Y_lag_6_pred = model_lag_6_xgb.predict(lag_6_test_m) # prédiction sur les données de test
Y_lag_6_pred_sc = scaler.inverse_transform(Y_lag_6_pred.reshape(-1, 1)) # reformatage de la prédiction
Y_lag_6_test_sc = scaler.inverse_transform(Y_lag_6_test.values.reshape(-1, 1)) # reformatage du Y de test

# calcul des variables de qualité
mae_lag_6 = mean_absolute_error(Y_lag_6_test_sc, Y_lag_6_pred_sc)
rmse_lag_6 = np.sqrt(mean_squared_error(Y_lag_6_test_sc, Y_lag_6_pred_sc))
r2_lag_6 = r2_score(Y_lag_6_test_sc, Y_lag_6_pred_sc)
mape_lag_6 = np.std(np.abs((Y_lag_6_test_sc - Y_lag_6_pred_sc) / Y_lag_6_test_sc)) * 100

# affichage de ces variables
print("MAE:", mae_lag_6)
print("RMSE:", rmse_lag_6)
print("R2:", r2_lag_6)
print("MAPE:", mape_lag_6)

# création du graphique pour observer le résultat
plt.figure(figsize=(12, 6))
plt.plot(Y_lag_6_test_sc, label='Vraies valeurs')
plt.plot(Y_lag_6_pred_sc, label='Prédictions')
plt.xlabel('Indice')
plt.ylabel('Consommation')
plt.title('XGBoost lag 6')
plt.legend()
plt.xlim([2000, 2167]) # toujours sur 1 semaine dans le test
plt.show()

# Pareil qu'avant mais avec un lag différent
print(" - XGBoost lag 12 - ")
lag_12_train_m = xgb.DMatrix(X_lag_12_train.reshape(-1, X_lag_12_train.shape[1]), label=Y_lag_12_train, enable_categorical=True)
lag_12_test_m = xgb.DMatrix(X_lag_12_test.reshape(-1, X_lag_12_test.shape[1]), label=Y_lag_12_test, enable_categorical=True)

model_lag_12_xgb = xgb.train(params={"objective":"reg:squarederror"}, dtrain=lag_12_train_m, num_boost_round=100)

Y_lag_12_pred = model_lag_12_xgb.predict(lag_12_test_m)
Y_lag_12_pred_sc = scaler.inverse_transform(Y_lag_12_pred.reshape(-1, 1))
Y_lag_12_test_sc = scaler.inverse_transform(Y_lag_12_test.values.reshape(-1, 1))

mae_lag_12 = mean_absolute_error(Y_lag_12_test_sc, Y_lag_12_pred_sc)
rmse_lag_12 = np.sqrt(mean_squared_error(Y_lag_12_test_sc, Y_lag_12_pred_sc))
r2_lag_12 = r2_score(Y_lag_12_test_sc, Y_lag_12_pred_sc)
mape_lag_12 = np.std(np.abs((Y_lag_12_test_sc - Y_lag_12_pred_sc) / Y_lag_12_test_sc)) * 100

print("MAE:", mae_lag_12)
print("RMSE:", rmse_lag_12)
print("R2:", r2_lag_12)
print("MAPE:", mape_lag_12)

plt.figure(figsize=(12, 6))
plt.plot(Y_lag_12_test_sc, label='Vraies valeurs')
plt.plot(Y_lag_12_pred_sc, label='Prédictions')
plt.xlabel('Indice')
plt.ylabel('Consommation')
plt.title('XGBoost lag 12')
plt.legend()
plt.xlim([2000, 2167])
plt.show()


print(" - XGBoost lag 24 - ")
lag_24_train_m = xgb.DMatrix(X_lag_24_train.reshape(-1, X_lag_24_train.shape[1]), label=Y_lag_24_train, enable_categorical=True)
lag_24_test_m = xgb.DMatrix(X_lag_24_test.reshape(-1, X_lag_24_test.shape[1]), label=Y_lag_24_test, enable_categorical=True)

model_lag_24_xgb = xgb.train(params={"objective":"reg:squarederror"}, dtrain=lag_24_train_m, num_boost_round=100)

Y_lag_24_pred = model_lag_24_xgb.predict(lag_24_test_m)
Y_lag_24_pred_sc = scaler.inverse_transform(Y_lag_24_pred.reshape(-1, 1))
Y_lag_24_test_sc = scaler.inverse_transform(Y_lag_24_test.values.reshape(-1, 1))

mae_lag_24 = mean_absolute_error(Y_lag_24_test_sc, Y_lag_24_pred_sc)
rmse_lag_24 = np.sqrt(mean_squared_error(Y_lag_24_test_sc, Y_lag_24_pred_sc))
r2_lag_24 = r2_score(Y_lag_24_test_sc, Y_lag_24_pred_sc)
mape_lag_24 = np.std(np.abs((Y_lag_24_test_sc - Y_lag_24_pred_sc) / Y_lag_24_test_sc)) * 100

print("MAE:", mae_lag_24)
print("RMSE:", rmse_lag_24)
print("R2:", r2_lag_24)
print("MAPE:", mape_lag_24)

plt.figure(figsize=(12, 6))
plt.plot(Y_lag_24_test_sc, label='Vraies valeurs')
plt.plot(Y_lag_24_pred_sc, label='Prédictions')
plt.xlabel('Indice')
plt.ylabel('Consommation')
plt.title('XGBoost lag 24')
plt.legend()
plt.xlim([2000, 2167])
plt.show()


# Pareil mais avec le rolling
print(" - XGBoost rol 6 mean - ")
rol_6_mean_train_m = xgb.DMatrix(X_rol_6_mean_train.reshape(-1, X_rol_6_mean_train.shape[1]), label=Y_rol_6_train, enable_categorical=True)
rol_6_mean_test_m = xgb.DMatrix(X_rol_6_mean_test.reshape(-1, X_rol_6_mean_test.shape[1]), label=Y_rol_6_test, enable_categorical=True)

model_rol_6_mean_xgb = xgb.train(params={"objective":"reg:squarederror"}, dtrain=rol_6_mean_train_m, num_boost_round=100)

Y_rol_6_mean_pred = model_rol_6_mean_xgb.predict(rol_6_mean_test_m)
Y_rol_6_mean_pred_sc = scaler.inverse_transform(Y_rol_6_mean_pred.reshape(-1, 1))
Y_rol_6_mean_test_sc = scaler.inverse_transform(Y_rol_6_test.values.reshape(-1, 1))

mae_rol_6_mean = mean_absolute_error(Y_rol_6_mean_test_sc, Y_rol_6_mean_pred_sc)
rmse_rol_6_mean = np.sqrt(mean_squared_error(Y_rol_6_mean_test_sc, Y_rol_6_mean_pred_sc))
r2_rol_6_mean = r2_score(Y_rol_6_mean_test_sc, Y_rol_6_mean_pred_sc)
mape_rol_6_mean = np.std(np.abs((Y_rol_6_mean_test_sc - Y_rol_6_mean_pred_sc) / Y_rol_6_mean_test_sc)) * 100

print("MAE:", mae_rol_6_mean)
print("RMSE:", rmse_rol_6_mean)
print("R2:", r2_rol_6_mean)
print("MAPE:", mape_rol_6_mean)

plt.figure(figsize=(12, 6))
plt.plot(Y_rol_6_mean_test_sc, label='Vraies valeurs')
plt.plot(Y_rol_6_mean_pred_sc, label='Prédictions')
plt.xlabel('Indice')
plt.ylabel('Consommation')
plt.title('XGBoost rol 6 mean')
plt.legend()
plt.xlim([2000, 2167])
plt.show()


print(" - XGBoost rol 12 mean - ")
rol_12_mean_train_m = xgb.DMatrix(X_rol_12_mean_train.reshape(-1, X_rol_12_mean_train.shape[1]), label=Y_rol_12_train, enable_categorical=True)
rol_12_mean_test_m = xgb.DMatrix(X_rol_12_mean_test.reshape(-1, X_rol_12_mean_test.shape[1]), label=Y_rol_12_test, enable_categorical=True)

model_rol_12_mean_xgb = xgb.train(params={"objective":"reg:squarederror"}, dtrain=rol_12_mean_train_m, num_boost_round=100)

Y_rol_12_mean_pred = model_rol_12_mean_xgb.predict(rol_12_mean_test_m)
Y_rol_12_mean_pred_sc = scaler.inverse_transform(Y_rol_12_mean_pred.reshape(-1, 1))
Y_rol_12_mean_test_sc = scaler.inverse_transform(Y_rol_12_test.values.reshape(-1, 1))

mae_rol_12_mean = mean_absolute_error(Y_rol_12_mean_test_sc, Y_rol_12_mean_pred_sc)
rmse_rol_12_mean = np.sqrt(mean_squared_error(Y_rol_12_mean_test_sc, Y_rol_12_mean_pred_sc))
r2_rol_12_mean = r2_score(Y_rol_12_mean_test_sc, Y_rol_12_mean_pred_sc)
mape_rol_12_mean = np.std(np.abs((Y_rol_12_mean_test_sc - Y_rol_12_mean_pred_sc) / Y_rol_12_mean_test_sc)) * 100

print("MAE:", mae_rol_12_mean)
print("RMSE:", rmse_rol_12_mean)
print("R2:", r2_rol_12_mean)
print("MAPE:", mape_rol_12_mean)

plt.figure(figsize=(12, 6))
plt.plot(Y_rol_12_mean_test_sc, label='Vraies valeurs')
plt.plot(Y_rol_12_mean_pred_sc, label='Prédictions')
plt.xlabel('Indice')
plt.ylabel('Consommation')
plt.title('XGBoost rol 12 mean')
plt.legend()
plt.xlim([2000, 2167])
plt.show()


print(" - XGBoost rol 24 mean - ")
rol_24_mean_train_m = xgb.DMatrix(X_rol_24_mean_train.reshape(-1, X_rol_24_mean_train.shape[1]), label=Y_rol_24_train, enable_categorical=True)
rol_24_mean_test_m = xgb.DMatrix(X_rol_24_mean_test.reshape(-1, X_rol_24_mean_test.shape[1]), label=Y_rol_24_test, enable_categorical=True)

model_rol_24_mean_xgb = xgb.train(params={"objective":"reg:squarederror"}, dtrain=rol_24_mean_train_m, num_boost_round=100)

Y_rol_24_mean_pred = model_rol_24_mean_xgb.predict(rol_24_mean_test_m)
Y_rol_24_mean_pred_sc = scaler.inverse_transform(Y_rol_24_mean_pred.reshape(-1, 1))
Y_rol_24_mean_test_sc = scaler.inverse_transform(Y_rol_24_test.values.reshape(-1, 1))

mae_rol_24_mean = mean_absolute_error(Y_rol_24_mean_test_sc, Y_rol_24_mean_pred_sc)
rmse_rol_24_mean = np.sqrt(mean_squared_error(Y_rol_24_mean_test_sc, Y_rol_24_mean_pred_sc))
r2_rol_24_mean = r2_score(Y_rol_24_mean_test_sc, Y_rol_24_mean_pred_sc)
mape_rol_24_mean = np.std(np.abs((Y_rol_24_mean_test_sc - Y_rol_24_mean_pred_sc) / Y_rol_24_mean_test_sc)) * 100

print("MAE:", mae_rol_24_mean)
print("RMSE:", rmse_rol_24_mean)
print("R2:", r2_rol_24_mean)
print("MAPE:", mape_rol_24_mean)

plt.figure(figsize=(12, 6))
plt.plot(Y_rol_24_mean_test_sc, label='Vraies valeurs')
plt.plot(Y_rol_24_mean_pred_sc, label='Prédictions')
plt.xlabel('Indice')
plt.ylabel('Consommation')
plt.title('XGBoost rol 24 mean')
plt.legend()
plt.xlim([2000, 2167])
plt.show()



print(" - XGBoost rol 6 min - ")
rol_6_min_train_m = xgb.DMatrix(X_rol_6_min_train.reshape(-1, X_rol_6_min_train.shape[1]), label=Y_rol_6_train, enable_categorical=True)
rol_6_min_test_m = xgb.DMatrix(X_rol_6_min_test.reshape(-1, X_rol_6_min_test.shape[1]), label=Y_rol_6_test, enable_categorical=True)

model_rol_6_min_xgb = xgb.train(params={"objective":"reg:squarederror"}, dtrain=rol_6_min_train_m, num_boost_round=100)

Y_rol_6_min_pred = model_rol_6_min_xgb.predict(rol_6_min_test_m)
Y_rol_6_min_pred_sc = scaler.inverse_transform(Y_rol_6_min_pred.reshape(-1, 1))
Y_rol_6_min_test_sc = scaler.inverse_transform(Y_rol_6_test.values.reshape(-1, 1))

mae_rol_6_min = mean_absolute_error(Y_rol_6_min_test_sc, Y_rol_6_min_pred_sc)
rmse_rol_6_min = np.sqrt(mean_squared_error(Y_rol_6_min_test_sc, Y_rol_6_min_pred_sc))
r2_rol_6_min = r2_score(Y_rol_6_min_test_sc, Y_rol_6_min_pred_sc)
mape_rol_6_min = np.std(np.abs((Y_rol_6_min_test_sc - Y_rol_6_min_pred_sc) / Y_rol_6_min_test_sc)) * 100

print("MAE:", mae_rol_6_min)
print("RMSE:", rmse_rol_6_min)
print("R2:", r2_rol_6_min)
print("MAPE:", mape_rol_6_min)

plt.figure(figsize=(12, 6))
plt.plot(Y_rol_6_min_test_sc, label='Vraies valeurs')
plt.plot(Y_rol_6_min_pred_sc, label='Prédictions')
plt.xlabel('Indice')
plt.ylabel('Consommation')
plt.title('XGBoost rol 6 min')
plt.legend()
plt.xlim([2000, 2167])
plt.show()


print(" - XGBoost rol 12 min - ")
rol_12_min_train_m = xgb.DMatrix(X_rol_12_min_train.reshape(-1, X_rol_12_min_train.shape[1]), label=Y_rol_12_train, enable_categorical=True)
rol_12_min_test_m = xgb.DMatrix(X_rol_12_min_test.reshape(-1, X_rol_12_min_test.shape[1]), label=Y_rol_12_test, enable_categorical=True)

model_rol_12_min_xgb = xgb.train(params={"objective":"reg:squarederror"}, dtrain=rol_12_min_train_m, num_boost_round=100)

Y_rol_12_min_pred = model_rol_12_min_xgb.predict(rol_12_min_test_m)
Y_rol_12_min_pred_sc = scaler.inverse_transform(Y_rol_12_min_pred.reshape(-1, 1))
Y_rol_12_min_test_sc = scaler.inverse_transform(Y_rol_12_test.values.reshape(-1, 1))

mae_rol_12_min = mean_absolute_error(Y_rol_12_min_test_sc, Y_rol_12_min_pred_sc)
rmse_rol_12_min = np.sqrt(mean_squared_error(Y_rol_12_min_test_sc, Y_rol_12_min_pred_sc))
r2_rol_12_min = r2_score(Y_rol_12_min_test_sc, Y_rol_12_min_pred_sc)
mape_rol_12_min = np.std(np.abs((Y_rol_12_min_test_sc - Y_rol_12_min_pred_sc) / Y_rol_12_min_test_sc)) * 100

print("MAE:", mae_rol_12_min)
print("RMSE:", rmse_rol_12_min)
print("R2:", r2_rol_12_min)
print("MAPE:", mape_rol_12_min)

plt.figure(figsize=(12, 6))
plt.plot(Y_rol_12_min_test_sc, label='Vraies valeurs')
plt.plot(Y_rol_12_min_pred_sc, label='Prédictions')
plt.xlabel('Indice')
plt.ylabel('Consommation')
plt.title('XGBoost rol 12 min')
plt.legend()
plt.xlim([2000, 2167])
plt.show()


print(" - XGBoost rol 24 min - ")
rol_24_min_train_m = xgb.DMatrix(X_rol_24_min_train.reshape(-1, X_rol_24_min_train.shape[1]), label=Y_rol_24_train, enable_categorical=True)
rol_24_min_test_m = xgb.DMatrix(X_rol_24_min_test.reshape(-1, X_rol_24_min_test.shape[1]), label=Y_rol_24_test, enable_categorical=True)

model_rol_24_min_xgb = xgb.train(params={"objective":"reg:squarederror"}, dtrain=rol_24_min_train_m, num_boost_round=100)

Y_rol_24_min_pred = model_rol_24_min_xgb.predict(rol_24_min_test_m)
Y_rol_24_min_pred_sc = scaler.inverse_transform(Y_rol_24_min_pred.reshape(-1, 1))
Y_rol_24_min_test_sc = scaler.inverse_transform(Y_rol_24_test.values.reshape(-1, 1))

mae_rol_24_min = mean_absolute_error(Y_rol_24_min_test_sc, Y_rol_24_min_pred_sc)
rmse_rol_24_min = np.sqrt(mean_squared_error(Y_rol_24_min_test_sc, Y_rol_24_min_pred_sc))
r2_rol_24_min = r2_score(Y_rol_24_min_test_sc, Y_rol_24_min_pred_sc)
mape_rol_24_min = np.std(np.abs((Y_rol_24_min_test_sc - Y_rol_24_min_pred_sc) / Y_rol_24_min_test_sc)) * 100

print("MAE:", mae_rol_24_min)
print("RMSE:", rmse_rol_24_min)
print("R2:", r2_rol_24_min)
print("MAPE:", mape_rol_24_min)

plt.figure(figsize=(12, 6))
plt.plot(Y_rol_24_min_test_sc, label='Vraies valeurs')
plt.plot(Y_rol_24_min_pred_sc, label='Prédictions')
plt.xlabel('Indice')
plt.ylabel('Consommation')
plt.title('XGBoost rol 24 min')
plt.legend()
plt.xlim([2000, 2167])
plt.show()



print(" - XGBoost rol 6 max - ")
rol_6_max_train_m = xgb.DMatrix(X_rol_6_max_train.reshape(-1, X_rol_6_max_train.shape[1]), label=Y_rol_6_train, enable_categorical=True)
rol_6_max_test_m = xgb.DMatrix(X_rol_6_max_test.reshape(-1, X_rol_6_max_test.shape[1]), label=Y_rol_6_test, enable_categorical=True)

model_rol_6_max_xgb = xgb.train(params={"objective":"reg:squarederror"}, dtrain=rol_6_max_train_m, num_boost_round=100)

Y_rol_6_max_pred = model_rol_6_max_xgb.predict(rol_6_max_test_m)
Y_rol_6_max_pred_sc = scaler.inverse_transform(Y_rol_6_max_pred.reshape(-1, 1))
Y_rol_6_max_test_sc = scaler.inverse_transform(Y_rol_6_test.values.reshape(-1, 1))

mae_rol_6_max = mean_absolute_error(Y_rol_6_max_test_sc, Y_rol_6_max_pred_sc)
rmse_rol_6_max = np.sqrt(mean_squared_error(Y_rol_6_max_test_sc, Y_rol_6_max_pred_sc))
r2_rol_6_max = r2_score(Y_rol_6_max_test_sc, Y_rol_6_max_pred_sc)
mape_rol_6_max = np.std(np.abs((Y_rol_6_max_test_sc - Y_rol_6_max_pred_sc) / Y_rol_6_max_test_sc)) * 100

print("MAE:", mae_rol_6_max)
print("RMSE:", rmse_rol_6_max)
print("R2:", r2_rol_6_max)
print("MAPE:", mape_rol_6_max)

plt.figure(figsize=(12, 6))
plt.plot(Y_rol_6_max_test_sc, label='Vraies valeurs')
plt.plot(Y_rol_6_max_pred_sc, label='Prédictions')
plt.xlabel('Indice')
plt.ylabel('Consommation')
plt.title('XGBoost rol 6 max')
plt.legend()
plt.xlim([2000, 2167])
plt.show()


print(" - XGBoost rol 12 max - ")
rol_12_max_train_m = xgb.DMatrix(X_rol_12_max_train.reshape(-1, X_rol_12_max_train.shape[1]), label=Y_rol_12_train, enable_categorical=True)
rol_12_max_test_m = xgb.DMatrix(X_rol_12_max_test.reshape(-1, X_rol_12_max_test.shape[1]), label=Y_rol_12_test, enable_categorical=True)

model_rol_12_max_xgb = xgb.train(params={"objective":"reg:squarederror"}, dtrain=rol_12_max_train_m, num_boost_round=100)

Y_rol_12_max_pred = model_rol_12_max_xgb.predict(rol_12_max_test_m)
Y_rol_12_max_pred_sc = scaler.inverse_transform(Y_rol_12_max_pred.reshape(-1, 1))
Y_rol_12_max_test_sc = scaler.inverse_transform(Y_rol_12_test.values.reshape(-1, 1))

mae_rol_12_max = mean_absolute_error(Y_rol_12_max_test_sc, Y_rol_12_max_pred_sc)
rmse_rol_12_max = np.sqrt(mean_squared_error(Y_rol_12_max_test_sc, Y_rol_12_max_pred_sc))
r2_rol_12_max = r2_score(Y_rol_12_max_test_sc, Y_rol_12_max_pred_sc)
mape_rol_12_max = np.std(np.abs((Y_rol_12_max_test_sc - Y_rol_12_max_pred_sc) / Y_rol_12_max_test_sc)) * 100

print("MAE:", mae_rol_12_max)
print("RMSE:", rmse_rol_12_max)
print("R2:", r2_rol_12_max)
print("MAPE:", mape_rol_12_max)

plt.figure(figsize=(12, 6))
plt.plot(Y_rol_12_max_test_sc, label='Vraies valeurs')
plt.plot(Y_rol_12_max_pred_sc, label='Prédictions')
plt.xlabel('Indice')
plt.ylabel('Consommation')
plt.title('XGBoost rol 12 max')
plt.legend()
plt.xlim([2000, 2167])
plt.show()


print(" - XGBoost rol 24 max - ")
rol_24_max_train_m = xgb.DMatrix(X_rol_24_max_train.reshape(-1, X_rol_24_max_train.shape[1]), label=Y_rol_24_train, enable_categorical=True)
rol_24_max_test_m = xgb.DMatrix(X_rol_24_max_test.reshape(-1, X_rol_24_max_test.shape[1]), label=Y_rol_24_test, enable_categorical=True)

model_rol_24_max_xgb = xgb.train(params={"objective":"reg:squarederror"}, dtrain=rol_24_max_train_m, num_boost_round=100)

Y_rol_24_max_pred = model_rol_24_max_xgb.predict(rol_24_max_test_m)
Y_rol_24_max_pred_sc = scaler.inverse_transform(Y_rol_24_max_pred.reshape(-1, 1))
Y_rol_24_max_test_sc = scaler.inverse_transform(Y_rol_24_test.values.reshape(-1, 1))

mae_rol_24_max = mean_absolute_error(Y_rol_24_max_test_sc, Y_rol_24_max_pred_sc)
rmse_rol_24_max = np.sqrt(mean_squared_error(Y_rol_24_max_test_sc, Y_rol_24_max_pred_sc))
r2_rol_24_max = r2_score(Y_rol_24_max_test_sc, Y_rol_24_max_pred_sc)
mape_rol_24_max = np.std(np.abs((Y_rol_24_max_test_sc - Y_rol_24_max_pred_sc) / Y_rol_24_max_test_sc)) * 100

print("MAE:", mae_rol_24_max)
print("RMSE:", rmse_rol_24_max)
print("R2:", r2_rol_24_max)
print("MAPE:", mape_rol_24_max)

plt.figure(figsize=(12, 6))
plt.plot(Y_rol_24_max_test_sc, label='Vraies valeurs')
plt.plot(Y_rol_24_max_pred_sc, label='Prédictions')
plt.xlabel('Indice')
plt.ylabel('Consommation')
plt.title('XGBoost rol 24 max')
plt.legend()
plt.xlim([2000, 2167])
plt.show()



print(" - XGBoost rol 6 std - ")
rol_6_std_train_m = xgb.DMatrix(X_rol_6_std_train.reshape(-1, X_rol_6_std_train.shape[1]), label=Y_rol_6_train, enable_categorical=True)
rol_6_std_test_m = xgb.DMatrix(X_rol_6_std_test.reshape(-1, X_rol_6_std_test.shape[1]), label=Y_rol_6_test, enable_categorical=True)

model_rol_6_std_xgb = xgb.train(params={"objective":"reg:squarederror"}, dtrain=rol_6_std_train_m, num_boost_round=100)

Y_rol_6_std_pred = model_rol_6_std_xgb.predict(rol_6_std_test_m)
Y_rol_6_std_pred_sc = scaler.inverse_transform(Y_rol_6_std_pred.reshape(-1, 1))
Y_rol_6_std_test_sc = scaler.inverse_transform(Y_rol_6_test.values.reshape(-1, 1))

mae_rol_6_std = mean_absolute_error(Y_rol_6_std_test_sc, Y_rol_6_std_pred_sc)
rmse_rol_6_std = np.sqrt(mean_squared_error(Y_rol_6_std_test_sc, Y_rol_6_std_pred_sc))
r2_rol_6_std = r2_score(Y_rol_6_std_test_sc, Y_rol_6_std_pred_sc)
mape_rol_6_std = np.std(np.abs((Y_rol_6_std_test_sc - Y_rol_6_std_pred_sc) / Y_rol_6_std_test_sc)) * 100

print("MAE:", mae_rol_6_std)
print("RMSE:", rmse_rol_6_std)
print("R2:", r2_rol_6_std)
print("MAPE:", mape_rol_6_std)

plt.figure(figsize=(12, 6))
plt.plot(Y_rol_6_std_test_sc, label='Vraies valeurs')
plt.plot(Y_rol_6_std_pred_sc, label='Prédictions')
plt.xlabel('Indice')
plt.ylabel('Consommation')
plt.title('XGBoost rol 6 std')
plt.legend()
plt.xlim([2000, 2167])
plt.show()


print(" - XGBoost rol 12 std - ")
rol_12_std_train_m = xgb.DMatrix(X_rol_12_std_train.reshape(-1, X_rol_12_std_train.shape[1]), label=Y_rol_12_train, enable_categorical=True)
rol_12_std_test_m = xgb.DMatrix(X_rol_12_std_test.reshape(-1, X_rol_12_std_test.shape[1]), label=Y_rol_12_test, enable_categorical=True)

model_rol_12_std_xgb = xgb.train(params={"objective":"reg:squarederror"}, dtrain=rol_12_std_train_m, num_boost_round=100)

Y_rol_12_std_pred = model_rol_12_std_xgb.predict(rol_12_std_test_m)
Y_rol_12_std_pred_sc = scaler.inverse_transform(Y_rol_12_std_pred.reshape(-1, 1))
Y_rol_12_std_test_sc = scaler.inverse_transform(Y_rol_12_test.values.reshape(-1, 1))

mae_rol_12_std = mean_absolute_error(Y_rol_12_std_test_sc, Y_rol_12_std_pred_sc)
rmse_rol_12_std = np.sqrt(mean_squared_error(Y_rol_12_std_test_sc, Y_rol_12_std_pred_sc))
r2_rol_12_std = r2_score(Y_rol_12_std_test_sc, Y_rol_12_std_pred_sc)
mape_rol_12_std = np.std(np.abs((Y_rol_12_std_test_sc - Y_rol_12_std_pred_sc) / Y_rol_12_std_test_sc)) * 100

print("MAE:", mae_rol_12_std)
print("RMSE:", rmse_rol_12_std)
print("R2:", r2_rol_12_std)
print("MAPE:", mape_rol_12_std)

plt.figure(figsize=(12, 6))
plt.plot(Y_rol_12_std_test_sc, label='Vraies valeurs')
plt.plot(Y_rol_12_std_pred_sc, label='Prédictions')
plt.xlabel('Indice')
plt.ylabel('Consommation')
plt.title('XGBoost rol 12 std')
plt.legend()
plt.xlim([2000, 2167])
plt.show()


print(" - XGBoost rol 24 std - ")
rol_24_std_train_m = xgb.DMatrix(X_rol_24_std_train.reshape(-1, X_rol_24_std_train.shape[1]), label=Y_rol_24_train, enable_categorical=True)
rol_24_std_test_m = xgb.DMatrix(X_rol_24_std_test.reshape(-1, X_rol_24_std_test.shape[1]), label=Y_rol_24_test, enable_categorical=True)

model_rol_24_std_xgb = xgb.train(params={"objective":"reg:squarederror"}, dtrain=rol_24_std_train_m, num_boost_round=100)

Y_rol_24_std_pred = model_rol_24_std_xgb.predict(rol_24_std_test_m)
Y_rol_24_std_pred_sc = scaler.inverse_transform(Y_rol_24_std_pred.reshape(-1, 1))
Y_rol_24_std_test_sc = scaler.inverse_transform(Y_rol_24_test.values.reshape(-1, 1))

mae_rol_24_std = mean_absolute_error(Y_rol_24_std_test_sc, Y_rol_24_std_pred_sc)
rmse_rol_24_std = np.sqrt(mean_squared_error(Y_rol_24_std_test_sc, Y_rol_24_std_pred_sc))
r2_rol_24_std = r2_score(Y_rol_24_std_test_sc, Y_rol_24_std_pred_sc)
mape_rol_24_std = np.std(np.abs((Y_rol_24_std_test_sc - Y_rol_24_std_pred_sc) / Y_rol_24_std_test_sc)) * 100

print("MAE:", mae_rol_24_std)
print("RMSE:", rmse_rol_24_std)
print("R2:", r2_rol_24_std)
print("MAPE:", mape_rol_24_std)

plt.figure(figsize=(12, 6))
plt.plot(Y_rol_24_std_test_sc, label='Vraies valeurs')
plt.plot(Y_rol_24_std_pred_sc, label='Prédictions')
plt.xlabel('Indice')
plt.ylabel('Consommation')
plt.title('XGBoost rol 24 std')
plt.legend()
plt.xlim([2000, 2167])
plt.show()
####
