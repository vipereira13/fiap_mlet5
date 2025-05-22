import streamlit as st
import requests
import pandas as pd
import plotly.express as px

#topo da pagina
st.set_page_config(page_title="RH Decision", layout="centered")
st.title("🤖 Avaliador automático de currículos - Decision")

#labels
vaga_texto = st.text_area("📄 Descrição da vaga", height=150)
uploaded_file = st.file_uploader("📎 Envie o currículo do candidato (PDF)", type=["pdf"])

#chamar API
if uploaded_file and vaga_texto:
    if st.button("🔍 Avaliar Candidato"):
        with st.spinner("Analisando dados..."):
            response = requests.post(
                "https://app-api-rh.onrender.com/predict/",
                data={"vaga": vaga_texto},
                files={"file": (uploaded_file.name, uploaded_file, "application/pdf")}
            )

        if response.status_code == 200:
            resultado = response.json()
            st.success(f"✅ Probabilidade de contratação: **{resultado['probabilidade_contratacao']}%**")
            st.markdown(f"**💡 Interpretação:** {resultado['interpretacao']}")

            exigidas = set(resultado["campos_extraidos"].get("keywords_exigidas", []))
            candidato = set(resultado["campos_extraidos"].get("keywords_candidato", []))



            st.subheader("📈 Métricas do modelo:")
            col1, col2, col3 = st.columns(3)
            col1.metric("Score Composto", f"{resultado.get('score_composto', 0)}%")
            col2.metric("Match Semântico", f"{resultado.get('semantic_match_score', 0)}")

            if exigidas:
                exigidas_list = sorted(list(exigidas))
                candidato_lower = set(map(str.lower, candidato))

                match_count = len([kw for kw in exigidas_list if kw.lower() in candidato_lower])
                total_exigidas = len(exigidas_list)
                percentual_match = round((match_count / total_exigidas) * 100, 1)
                col3.metric("Aderência técnica a vaga", f"{percentual_match}%")

                # Tabela abaixo das métricas
                st.markdown("### 📋 Tabela de correspondência de palavras-chave")
                df_keywords = pd.DataFrame({
                    "Keyword": exigidas_list,
                    "Match": ["Sim" if kw.lower() in candidato_lower else "Não" for kw in exigidas_list]
                })
                st.dataframe(df_keywords, use_container_width=True)

                # Gráfico abaixo da tabela
                fig_kw = px.bar(df_keywords, x="Keyword", color="Match",
                                color_discrete_map={"Sim": "green", "Não": "red"},
                                title="🔎 Visualização de Match das Keywords Técnicas",
                                labels={"Keyword": "Palavra-chave", "Match": "Presente no currículo?"})
                fig_kw.update_layout(xaxis_title="Keyword", yaxis_title="Presença", showlegend=True)
                st.plotly_chart(fig_kw, use_container_width=True)

            else:
                st.warning("⚠️ Nenhuma keyword exigida foi detectada para esta vaga.")

            st.subheader("🧭 Perfil cultural geral:")
            st.metric("Nota", resultado.get("perfil_cultural", "N/A"))
            st.markdown(f" **📝 Descrição:** {resultado.get('descricao_cultural', 'Sem descrição disponível.')}")

            st.subheader("📌 Campos extraídos do currículo:")
            st.json(resultado["campos_extraidos"])




            perfil_detalhado = resultado.get("perfil_cultural_detalhado")
            if isinstance(perfil_detalhado, dict):
                radar_data = pd.DataFrame({
                    "dimensao": list(perfil_detalhado.keys()),
                    "nota": list(perfil_detalhado.values())
                })
                fig = px.line_polar(radar_data, r="nota", theta="dimensao", line_close=True,
                                    title="Radar de Perfil Cultural", range_r=[0, 10])
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("❌ Erro ao processar a requisição:")
            st.code(response.text)
else:
    st.info("📝 Preencha a descrição da vaga e envie o currículo do candidato.")