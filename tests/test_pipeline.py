
import pytest
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import StandardScaler
import joblib

@pytest.fixture
def sample_data():
    return {
        "nivel_ingles": "Intermediário",
        "nivel_espanhol": "Básico",
        "nivel_academico": "Ensino Superior Completo",
        "vaga_sap": "Não",
        "match_score": 8,
        "keywords_exigidas": ["Kotlin", "API", "SQL"],
        "keywords_candidato": ["Kotlin", "API", "SQL", "Firebase"],
        "perfil_cultural": 7,
    }

def test_encoding(sample_data):
    idioma_niveis = joblib.load("models/encoder_idiomas.joblib")
    encoder_academico = joblib.load("models/encoder_academico.joblib")

    assert idioma_niveis["Intermediário"] == 2
    assert idioma_niveis["Básico"] == 1
    assert encoder_academico["Ensino Superior Completo"] == 7

def test_semantic_similarity(sample_data):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb_vaga = model.encode(sample_data["keywords_exigidas"], convert_to_tensor=True)
    emb_cv = model.encode(sample_data["keywords_candidato"], convert_to_tensor=True)
    sim_matrix = util.pytorch_cos_sim(emb_vaga, emb_cv)
    similarity = float(sim_matrix.max(dim=1).values.mean())
    assert 0 <= similarity <= 1

def test_scaler():
    scaler = joblib.load("models/scaler.joblib")
    X = pd.DataFrame([[2, 1, 7, 0, 0.95, 8, 1, 4, 1, 0.92]],
                     columns=["nivel_ingles", "nivel_espanhol", "nivel_academico", "vaga_sap",
                              "semantic_match_score", "match_score", "gap_ingles", "gap_nivel",
                              "tem_sap", "score_composto"])
    X_scaled = scaler.transform(X)
    assert X_scaled.shape == (1, 10)
