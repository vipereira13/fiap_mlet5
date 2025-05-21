

import json
import pandas as pd
import re
import os

def extract_tech_keywords(text):
    if not isinstance(text, str):
        return []

    tech_terms = [
        "SAP", "SAP MM", "SAP SD", "SAP FI", "SAP CO", "SAP PP", "SAP WM", "SAP BASIS", "SQL", "Oracle",
        "Python", "Java", "ABAP", "S4/HANA", "ECC", "Rollout", "Blueprint", "Go Live", "AMS", "Metodologia Ágil",
        "Scrum", "Kanban", "DevOps", "Cloud", "AWS", "GCP", "Azure", "BI", "Data Warehouse", "ETL", "CRM",
        "ERP", "API", "REST", "SOAP", "Machine Learning", "Power BI", "Excel Avançado", "PMO"
    ]
    found_terms = [term for term in tech_terms if re.search(rf'\\b{re.escape(term)}\\b', text, re.IGNORECASE)]
    return list(set(found_terms))

def load_and_prepare_data():

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_path = os.path.join(BASE_DIR, "data/")

    with open(base_path + 'vagas.json', 'r', encoding='utf-8') as f:
        vagas = json.load(f)

    with open(base_path + 'prospects.json', 'r', encoding='utf-8') as f:
        prospects = json.load(f)

    with open(base_path + 'applicants.json', 'r', encoding='utf-8') as f:
        applicants = json.load(f)

    rows = []
    for vaga_id, vaga in prospects.items():
        if vaga_id not in vagas:
            continue
        vaga_info = vagas[vaga_id]
        requisitos = vaga_info.get("perfil_vaga", {})
        info_basica = vaga_info.get("informacoes_basicas", {})

        for candidato in vaga.get("prospects", []):
            codigo_candidato = candidato.get("codigo")
            dados_candidato = applicants.get(codigo_candidato, {})
            if not dados_candidato:
                continue

            formacao = dados_candidato.get("formacao_e_idiomas", {})
            profissional = dados_candidato.get("informacoes_profissionais", {})
            cv_texto = dados_candidato.get("cv_pt", "")

            row = {
                "codigo_vaga": vaga_id,
                "titulo_vaga": info_basica.get("titulo_vaga"),
                "cliente": info_basica.get("cliente"),
                "sap": info_basica.get("vaga_sap"),
                "nivel_profissional_exigido": requisitos.get("nivel profissional"),
                "nivel_academico_exigido": requisitos.get("nivel_academico"),
                "nivel_ingles_exigido": requisitos.get("nivel_ingles"),
                "nivel_espanhol_exigido": requisitos.get("nivel_espanhol"),
                "competencias_tecnicas_exigidas": requisitos.get("competencia_tecnicas_e_comportamentais"),
                "codigo_candidato": codigo_candidato,
                "nome_candidato": candidato.get("nome"),
                "situacao_candidato": candidato.get("situacao_candidado"),
                "nivel_academico": formacao.get("nivel_academico"),
                "nivel_ingles": formacao.get("nivel_ingles"),
                "nivel_espanhol": formacao.get("nivel_espanhol"),
                "outro_idioma": formacao.get("outro_idioma"),
                "conhecimentos_tecnicos": profissional.get("conhecimentos_tecnicos"),
                "cv_pt": cv_texto,
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    df["target_contratado"] = df["situacao_candidato"].str.lower().apply(
        lambda x: 1 if "contratado" in x else 0
    )

    for col in ["nivel_academico", "nivel_ingles", "nivel_espanhol"]:
        df[col] = df.apply(
            lambda row: row[col] if row[col] and isinstance(row[col], str) and row[col].strip() else (
                re.search(rf"{col.replace('_', ' ')}:?\s*([^]*)", row["cv_pt"], re.IGNORECASE).group(1).strip()
                if re.search(rf"{col.replace('_', ' ')}:?\s*(.*)", row["cv_pt"], re.IGNORECASE)

                else ""
            ),
            axis=1
        )

    df["keywords_exigidas"] = df["competencias_tecnicas_exigidas"].apply(extract_tech_keywords)
    df["keywords_candidato"] = df["cv_pt"].apply(extract_tech_keywords)

    df["match_keywords"] = df.apply(
        lambda row: list(set(row["keywords_exigidas"]).intersection(set(row["keywords_candidato"]))),
        axis=1
    )

    df["match_score"] = df["match_keywords"].apply(len)

    return df




df_teste = load_and_prepare_data()