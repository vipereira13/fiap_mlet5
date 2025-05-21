from data_loader import load_and_prepare_data

import pandas as pd
import numpy as np
import os
import ast
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from xgboost import XGBClassifier
from imblearn.combine import SMOTETomek
from sentence_transformers import SentenceTransformer, util

# Carregar base
df = load_and_prepare_data()

# Função para avaliar listas
def safe_literal_eval(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except:
            return []
    return x if isinstance(x, list) else []

# Aplicar nas colunas de keyword
df["keywords_exigidas"] = df["keywords_exigidas"].apply(safe_literal_eval)
df["keywords_candidato"] = df["keywords_candidato"].apply(safe_literal_eval)

# Modelo de embeddings
model_emb = SentenceTransformer("all-MiniLM-L6-v2")

def calc_match_score_semantic(row):
    exigidas = row["keywords_exigidas"]
    candidato = row["keywords_candidato"]
    if not exigidas or not candidato:
        return 0.0
    emb_exigidas = model_emb.encode(exigidas, convert_to_tensor=True)
    emb_candidato = model_emb.encode(candidato, convert_to_tensor=True)
    sim_matrix = util.pytorch_cos_sim(emb_exigidas, emb_candidato)
    return float(sim_matrix.max(dim=1).values.mean())

df["semantic_match_score"] = df.apply(calc_match_score_semantic, axis=1)

# Codificações
idioma_niveis = {"Nenhum": 0, "Básico": 1, "Intermediário": 2, "Avançado": 3, "Fluente": 4}
for col in ["nivel_ingles", "nivel_espanhol"]:
    df[col] = df[col].map(idioma_niveis).fillna(0)

academico_niveis = {
    "Fundamental Incompleto": 0, "Fundamental Completo": 1, "Médio Incompleto": 2, "Médio Completo": 3,
    "Técnico": 4, "Tecnólogo": 5, "Superior Incompleto": 6, "Ensino Superior Completo": 7,
    "Pós-graduação": 8, "MBA": 9, "Mestrado": 10, "Doutorado": 11
}
df["nivel_academico"] = df["nivel_academico"].map(academico_niveis).fillna(0)
df["vaga_sap"] = df["sap"].map({"Sim": 1, "Não": 0}).fillna(0)

# Feature engineering
df["gap_ingles"] = df["nivel_ingles"] - 1
df["gap_nivel"] = df["nivel_academico"] - 3
df["tem_sap"] = df["keywords_candidato"].astype(str).str.contains("sap", case=False).astype(int)
df["score_composto"] = 0  # não é usado no treino, mas placeholder

# Definir features - Seria interessante a decision mapear o perfil cultural de toda a base de dados para inclusão
# da feature no modelo
features = ["nivel_ingles", "nivel_espanhol", "nivel_academico", "vaga_sap",
            "semantic_match_score", "match_score", "gap_ingles", "gap_nivel", "tem_sap", "score_composto"]
X = df[features]
y = df["target_contratado"]

# Balanceamento
X_res, y_res = SMOTETomek().fit_resample(X, y)

# Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_res)

# Modelo e avaliação
model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_scaled, y_res, cv=cv, scoring="f1_weighted")
print(f"F1 médio: {scores.mean():.4f}")

# Treinamento final
model.fit(X_scaled, y_res)
y_pred = model.predict(X_scaled)
print(confusion_matrix(y_res, y_pred))
print(classification_report(y_res, y_pred))

# Identificar melhor threshold
probas = model.predict_proba(X_scaled)[:, 1]
best_t, best_f1 = 0, 0
for t in np.arange(0.1, 0.9, 0.01):
    preds = (probas >= t).astype(int)
    f1 = f1_score(y_res, preds)
    if f1 > best_f1:
        best_f1, best_t = f1, t

print(f"Melhor threshold: {best_t:.2f} | F1-score: {best_f1:.4f}")

# Salvar modelo em joblib
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.joblib")
joblib.dump(scaler, "models/scaler.joblib")
joblib.dump(academico_niveis, "models/encoder_academico.joblib")
joblib.dump(idioma_niveis, "models/encoder_idiomas.joblib")
joblib.dump(best_t, "models/threshold_otimo.joblib")