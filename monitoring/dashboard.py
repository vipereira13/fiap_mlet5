import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Monitoramento do Modelo", layout="wide")
st.title("Painel de Monitoramento")

LOG_FILE = "logs/drift_monitoring.csv"

if os.path.exists(LOG_FILE):
    df = pd.read_csv(LOG_FILE)

    st.markdown("###  Últimas Inferências")
    st.dataframe(df.tail(20), use_container_width=True)

    st.markdown("###  Probabilidade ao Longo do Tempo")
    fig = px.line(df, x="data", y="probabilidade", markers=True, title="Evolução das Probabilidades")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("###  Distribuição das Previsões")
    fig2 = px.histogram(df, x="probabilidade", nbins=20, color=df["contratado_predito"].astype(str),
                        labels={"color": "Contratado Predito"},
                        title="Distribuição das Probabilidades de Contratação")
    st.plotly_chart(fig2, use_container_width=True)

    if "contratado_real" in df.columns and df["contratado_real"].isin([0, 1]).any():
        st.markdown("### 📊 Comparação Real vs Predito")
        df_valid = df[df["contratado_real"].isin([0, 1])]
        cm = pd.crosstab(df_valid["contratado_real"], df_valid["contratado_predito"],
                         rownames=["Real"], colnames=["Predito"])
        st.dataframe(cm)
else:
    st.warning("⚠️ Nenhuma inferência registrada ainda. Faça uma predição na API para ativar o monitoramento.")
