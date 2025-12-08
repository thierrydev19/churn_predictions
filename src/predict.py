from http import client
import pandas as pd
import joblib
import argparse
import yaml
import os   
import mlflow 
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import logging
from sklearn.preprocessing import LabelEncoder

def setup_logger(log_path):
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def preprocess_for_inference(df, encoders, scaler):
    #supprimer les identifiants inutiles
    df = df.drop(columns=['customerID'], errors='ignore')
    #colonnes numériques à convertir et nettoyer
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())
    #encodage des colonnes catégorielles avec les encodeurs sauvegardés

    for col, encoder in encoders.items():
        df[col] = df[col].astype(str).fillna("Unknown")
        df[col] = df[col].apply(lambda x: x if x in encoder.classes_ else "Unknown")
        df[col] = encoder.transform(df[col])

    #mise à l'échelle des données scaler les features
    return scaler.transform(df)



def preprocess_input_data(df):
    for col in df.select_dtypes(include='number').columns:
        df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    return df

def main(config_path):
    #chargement de la configuration et initialisation du logger
    config = load_config(config_path)
    setup_logger(config['logging']['log_file_predict'])
    logging.info("Starting prediction process.")

    #chemins des fichiers et des paramètres depuis la configuration
    input_path = config['predict']['input_path']
    output_path = config['predict']['output_path']
    model_registry_name = config['predict']['model_registry_name']
    model_version = config['predict']['version_to_use']
    tracking_uri = config['mlflow']['tracking_uri']
    scaler_path = config['model']['scaler_path']
    label_encoders_path = config['model']['label_encoders_path']


    #chargement des données brutes à prédire
    df_raw = pd.read_csv(input_path)
    logging.info(f"données brutes chargées depuis input {input_path} with shape {df_raw.shape}.")

    #connexion et chargement du modèle depuis MLflow
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{model_registry_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri) #permet de recharger le modèle enregistré
    logging.info(f"Modèle chargé depuis MLflow avec URI: {model_uri}.")

    # recuperer le run_id du modèle enregistré
    client = MlflowClient()
    model_version_info = client.get_model_version(name=model_registry_name, version=model_version)
    source_run_id = model_version_info.run_id
    logging.info(f"Run ID for the model version {model_version}: {source_run_id}")

    #télécharger les artéfacts preprocessing
    preprocessing_dir = "preprocessing_artifacts"
    os.makedirs(preprocessing_dir, exist_ok=True)
    
    logging.info(f"Téléchargement des artéfacts de prétraitement dans {preprocessing_dir}.")

    client.download_artifacts(source_run_id, "preprocessing/scaler.pkl", preprocessing_dir)
    client.download_artifacts(source_run_id, "preprocessing/label_encoders.pkl", preprocessing_dir)


   
# chargement du scaler et des encodeurs récupérés depuis mlflow
    scaler = joblib.load(os.path.join(preprocessing_dir, "scaler.pkl"))
    encoders = joblib.load(os.path.join(preprocessing_dir,"label_encoders.pkl"))

# prétraitement des nouvelles données 
    df_cleaned = preprocess_for_inference(df_raw.copy(), encoders, scaler)
    logging.info("Prétraitement des nouvelles données effectué.")

# prediction 
    y_pred = model.predict(df_cleaned)
    df_raw['predicted_churn'] = y_pred
# sauvegarde des résultats  
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_raw.to_csv(output_path, index=False)
    logging.info(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    args = parser.parse_args()
main(args.config)