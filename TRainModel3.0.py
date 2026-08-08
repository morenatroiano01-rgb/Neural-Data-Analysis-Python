import pandas as pd #gestione dati cvs
import numpy as np  #gestione array
from sklearn.model_selection import train_test_split #logica di divisione del dataset (X e y)
from sklearn.ensemble import RandomForestClassifier  #logica di apprendimento (alberi decisionali)
from sklearn.metrics import classification_report, accuracy_score #valutazioni: percentuali previsioni corrette, report dettagliato
from sklearn.preprocessing import RobustScaler #aggiungo un robust scaler per far lavorare il train model sulla mediana e non sulla media inmodo da ignorarei picchi assurdi del total power (artefatti muscolari)
import joblib #salvataggio modello
import os

DATASET= r"C:\Users\Morena\Documents\VScode\venv\BCI2.0_Project\dataset_eeg_preprocessed.csv"
MODELLO= r"C:\Users\Morena\Documents\VScode\venv\BCI_Project\Random_forest3.joblib"
SCALER_PATH = r"C:\Users\Morena\Documents\VScode\venv\BCI_Project\scaler_rf3.joblib"
#controllo di sicurezza per i dati
print(f"Caricamento {DATASET}... ")
if not os.path.exists(DATASET):
    raise FileNotFoundError ("!!FILE NON TROVATO!!")

df= pd.read_csv(DATASET) #carico i dati nella variabile DataFrame
df_clean= df[df['label'].isin([0, 1])].copy() #puliamo le colonne etichettate come -1

n_total= len(df)
n_valid= len(df_clean)
if n_valid== 0 :
    raise ValueError("Errore Critico: Nesun valore trovato (0, 1) ")
print(f"Campioni Totali: {n_total}   Campioni Validi: {n_valid}")

FEATURES = ['mean','std','alpha_power','beta_power', #colonna delle features nel data frame
            'theta_power','beta_alpha_ratio', 'dominant_alpha_freq', 
            'dominant_beta_freq', 'dominant_theta_freq', 'total_power']   
TARGET='label'                #colonna delle etichette nel data frame
X=df_clean[FEATURES]
y=df_clean[TARGET]
#APPLICO LA LOGICA DELLA MEDIANA
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X) # Applica la logica della mediana

#Divisione Training e Testing
X_train, X_test, y_train, y_test= train_test_split(X_scaled, y, test_size=0.2 , random_state=42, stratify=y)
#divido i dati X, y in dati di train e test, random_state assicura che la divisione casuale 
#si ariproducibile (se rieseguo lo script la divisione sarà sempre la stessa, per confrontare meglio i risultati)
print("----DIVISIONE DATI-----")
print(f"Training set 80% : {len(X_train)}")
print(f"Test set 20% : {len(X_test)}")

#Addestramento Modello
rf_model=RandomForestClassifier( #assegno l'oggetto RFC alla variabile
n_estimators=100, #numero di alberi decisionali della foresta
random_state= 42, #come prima per riproducibilità (costruisce i 100 alberi sempre allo stesso modo)
class_weight= 'balanced', #durante l'addestramento se i dati sono sbilanciati (più 1) cerca di dare più importanza alla classe che ha meno campioni
n_jobs=-1 )#uso tutti i core della cpu per velocizzare

rf_model.fit(X_train, y_train)
print("Addestramento Modello Completato con successo.")

#Valutazione delle performance
print("Valutazione modello sulla base di dati mai visti:")

y_pred=rf_model.predict(X_test) #gli do in pasto le features di test

#Report di classificazione
report=classification_report( #assegno l'oggetto classificatore alla variabile
    y_test, y_pred, target_names=['Rilassato (0)', 'Concentrato (1)'] 
)
accuratezza= accuracy_score(y_test, y_pred) #definisco l'accuratezza del odello
rilevanza_features=pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values(ascending=False) #definisco quali feature sono stati più rilevanti per una corretta predizione

print(f"Accuratezza sul set di test: {accuratezza:.4f}")#solo 4 cifre decimale
print("\nReport di Classificazione Dettagliato :\n", report)
print("\nImportanza delle Feature :\n", rilevanza_features.to_string())

#salvataggio del modello 
joblib.dump(rf_model, MODELLO)
joblib.dump(scaler, SCALER_PATH)
print(f"\n✅Modello salvato correttamente come '{MODELLO}' ")
print(f"✅ Scaler salvato correttamente in: {SCALER_PATH}")