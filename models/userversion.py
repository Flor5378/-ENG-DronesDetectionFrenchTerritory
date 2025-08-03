import pandas as pd
import numpy as np
import joblib
import os
import tkinter as tk
from tkinter import simpledialog, messagebox
from collections import Counter

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# --------------------
# 1. Chemins
# --------------------
chemin_models = r"C:\Users\fgrol\Documents\stages\models"
chemin_csv = r"C:\Users\fgrol\Documents\stages\csv_test.csv"

# --------------------
# 2. Charger les modèles
# --------------------
encoder = joblib.load(os.path.join(chemin_models, "label_encoder.pkl"))
knn_model = joblib.load(os.path.join(chemin_models, "knn_model.pkl"))
rf_model = joblib.load(os.path.join(chemin_models, "rf_model.pkl"))
catboost_model = joblib.load(os.path.join(chemin_models, "catboost_model.pkl"))
svm_model = joblib.load(os.path.join(chemin_models, "svm_model.pkl"))

# --------------------
# 3. Colonnes features
# --------------------
colonnes_features = [
    'x', 'y', 'altitude', 'hauteur', 'vitesse',
    'vitesse_calculee', 'delta_t', 'vz',
    'distance', 'temps_total', 'distance_totale',
    'longitude_depart'
]

# --------------------
# 4. Charger et nettoyer les données
# --------------------
df = pd.read_csv(chemin_csv, sep=',', quotechar='"', engine='python', nrows=10000)
X = df[colonnes_features].apply(pd.to_numeric, errors='coerce')
X = X.replace([np.inf, -np.inf], np.nan).dropna()

# --------------------
# 5. Prédictions
# --------------------
pred_knn = encoder.inverse_transform(knn_model.predict(X))
pred_rf = encoder.inverse_transform(rf_model.predict(X))
pred_cb = encoder.inverse_transform(catboost_model.predict(X))
pred_svm = encoder.inverse_transform(svm_model.predict(X))

df_predictions = df.loc[X.index].copy()
df_predictions['KNN'] = pred_knn
df_predictions['RandomForest'] = pred_rf
df_predictions['CatBoost'] = pred_cb
df_predictions['SVM'] = pred_svm

# --------------------
# 6. Vote majoritaire par ligne
# --------------------
def vote_majoritaire(row):
    votes = [row['KNN'], row['RandomForest'], row['CatBoost'], row['SVM']]
    return Counter(votes).most_common(1)[0][0]

df_predictions['Prediction_finale'] = df_predictions.apply(vote_majoritaire, axis=1)




# --------------------
# 7. Vote majoritaire par drone
# --------------------
df_par_drone = df_predictions.groupby('drone_id')[['KNN', 'RandomForest', 'CatBoost', 'Prediction_finale']] \
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 'Inconnu') \
    .reset_index()

# --------------------
# 8. Interface utilisateur
# --------------------
root = tk.Tk()
root.withdraw()

drone_id_input = simpledialog.askstring("Drone à prédire", "Entrez un ID de drone :")

if drone_id_input is None:
    messagebox.showinfo("Annulé", "Aucune entrée fournie. Fin du programme.")
    exit()

resultat = df_par_drone[df_par_drone['drone_id'] == drone_id_input]

if resultat.empty:
    messagebox.showerror("Erreur", f"Drone ID '{drone_id_input}' non trouvé.")
else:
    row = resultat.iloc[0]
    message = (
        f"📌 Drone ID : {row['drone_id']}\n\n"
        f"✅ Prediction finale : {row['Prediction_finale']}\n"
        f"🔍 KNN           : {row['KNN']}\n"
        f"🌲 Random Forest : {row['RandomForest']}\n"
        f"🐈 CatBoost      : {row['CatBoost']}"
    )

    messagebox.showinfo("Résultat de la prédiction", message)
