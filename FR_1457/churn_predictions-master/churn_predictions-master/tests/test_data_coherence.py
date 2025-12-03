import pandas as pd
import pytest

def test_data_integrity():
    path = 'data/processed/clean_data.csv'
    df = pd.read_csv(path)

    # Vérifie qu’il n’y a pas de valeurs manquantes
    assert not df.isnull().values.any(), "Le jeu de données contient des valeurs manquantes."

    # Vérifie la présence de la colonne cible
    assert 'churn' in df.columns, "'churn' n'est pas présent dans les colonnes."

    # Vérifie qu’il y a assez de colonnes pour entraîner un modèle
    assert df.shape[1] >= 2, "Le jeu de données a trop peu de colonnes."
