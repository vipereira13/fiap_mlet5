from fastapi import FastAPI, File, UploadFile, Form, Query
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer, util
import joblib
import pandas as pd
import pdfplumber
import json
from openai import OpenAI
from dotenv import load_dotenv
import os
from monitoring.monitoring import registrar_inferencia

app = FastAPI()

# Carregar modelos e encoders
model = joblib.load("models/model.joblib")
scaler = joblib.load("models/scaler.joblib")
encoder_academico = joblib.load("models/encoder_academico.joblib")
idioma_niveis = joblib.load("models/encoder_idiomas.joblib")
threshold_otimo = joblib.load("models/threshold_otimo.joblib")

# Modelo de embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Função para extrair texto do PDF
def extrair_texto_pdf(file: UploadFile):
    with pdfplumber.open(file.file) as pdf:
        texto = "".join([page.extract_text() or "" for page in pdf.pages])
    return texto.strip()

# Cálculo de similaridade
def calcular_match_semantico(keywords_vaga, keywords_cv):
    if not keywords_vaga or not keywords_cv:
        return 0.0
    emb_vaga = embedding_model.encode(keywords_vaga, convert_to_tensor=True)
    emb_cv = embedding_model.encode(keywords_cv, convert_to_tensor=True)
    sim_matrix = util.pytorch_cos_sim(emb_vaga, emb_cv)
    return float(sim_matrix.max(dim=1).values.mean())

#Carregar env
load_dotenv()

#criar client da openai
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


#Funcão para consultar o currículo e descrição da vaga no LLM
def consultar_llm(texto_cv, vaga):
    prompt = f"""
Você é um especialista em Recrutamento e Seleção de talentos da área de tecnologia. 

Seu objetivo é analisar o currículo abaixo com base na vaga descrita e extrair os campos solicitados.

Foque em identificar palavras-chave técnicas reais como linguagens, ferramentas, frameworks, tecnologias de backend/frontend, metodologias ágeis, bancos de dados, bibliotecas, APIs, IDEs e plataformas.

Use como referência tanto os títulos quanto os verbos de ação do currículo, especialmente em tópicos como "experiência", "tecnologias utilizadas", "responsável por", "projetos" etc.

Responda com os campos abaixo em JSON válido:

- nivel_ingles
- nivel_espanhol
- outro_idioma
- nivel_academico
- match_score (0 a 10)
- vaga_sap (Sim ou Não)
- keywords_exigidas
- keywords_candidato
- perfil_cultural (0 a 10)
- descricao_cultural (máx. 30 palavras)

VAGA:
{vaga}

CURRÍCULO:
\"\"\"{texto_cv}\"\"\"
"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000
        )
        resposta_texto = response.choices[0].message.content
        return json.loads(resposta_texto)
    except Exception as e:
        return {"erro": str(e)}


#funcão para preparar o input dos dados no modelo
def preparar_input(dados):
    df = pd.DataFrame([dados])

    for col in ["nivel_ingles", "nivel_espanhol"]:
        df[col] = df[col].map(idioma_niveis).fillna(0)

    df["nivel_academico"] = df["nivel_academico"].map(encoder_academico).fillna(0)
    df["vaga_sap"] = 1 if str(dados.get("vaga_sap", "")).strip().lower() == "sim" else 0
    df["semantic_match_score"] = calcular_match_semantico(
        dados.get("keywords_exigidas", []),
        dados.get("keywords_candidato", [])
    )
    df["match_score"] = float(dados.get("match_score", 0))
    df["gap_ingles"] = df["nivel_ingles"] - 1
    df["gap_nivel"] = df["nivel_academico"] - 3
    df["tem_sap"] = str(dados.get("keywords_candidato", "")).lower().count("sap")
    df["score_composto"] = 0  # modelo agora o aprende

    final_features = [
        "nivel_ingles", "nivel_espanhol", "nivel_academico", "vaga_sap",
        "semantic_match_score", "match_score", "gap_ingles", "gap_nivel", "tem_sap", "score_composto"
    ]
    return df, scaler.transform(df[final_features])

@app.post("/predict/")
async def predict(
    file: UploadFile = File(...),
    vaga: str = Form(...),
    threshold: float = Query(None, ge=0.0, le=1.0)
):
    texto_cv = extrair_texto_pdf(file)
    dados_extraidos = consultar_llm(texto_cv, vaga)

    if "erro" in dados_extraidos:
        return JSONResponse(status_code=400, content=dados_extraidos)

    try:
        df_input, input_array = preparar_input(dados_extraidos)
        prob = model.predict_proba(input_array)[0][1]
        usado_threshold = threshold if threshold is not None else threshold_otimo
        contratado = prob >= usado_threshold

        # Calcular score composto
        match_score = df_input["match_score"].values[0] / 10
        semantic = df_input["semantic_match_score"].values[0]
        cultural = int(dados_extraidos.get("perfil_cultural", 0)) / 10
        gap_ingles = df_input["gap_ingles"].values[0]
        bonus_ingles = 0.1 if gap_ingles >= 1 else 0
        bonus_cultural = 0.1 if cultural >= 0.8 else 0
        score_composto = min(1.0, 0.4 * match_score + 0.4 * semantic + 0.2 * cultural + bonus_ingles + bonus_cultural)

        # Interpretação
        if prob >= 0.8:
            interpretacao = "Perfil Ideal"
        elif prob >= 0.5:
            interpretacao = "Muito promissor"
        elif prob >= usado_threshold:
            interpretacao = "Pode ser avaliado"
        else:
            interpretacao = "Baixa probabilidade"

        registrar_inferencia(prob, contratado)

        return {
            "probabilidade_contratacao": float(round(prob * 100, 2)),
            "score_composto": float(round(score_composto * 100, 2)),
            "semantic_match_score": float(round(df_input["semantic_match_score"].values[0], 3)),
            "contratado_predito": bool(contratado),
            "interpretacao": interpretacao,
            "threshold_utilizado": float(usado_threshold),
            "campos_extraidos": dados_extraidos,
            "perfil_cultural": int(dados_extraidos.get("perfil_cultural", 0)),
            "descricao_cultural": dados_extraidos.get("descricao_cultural"),
            "perfil_cultural_detalhado": dados_extraidos.get("perfil_cultural_detalhado")
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"erro": str(e)})