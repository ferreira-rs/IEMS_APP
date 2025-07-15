# ---------------- IMPORTAÇÃO DE PACOTES ------------------
import streamlit as st
import pandas as pd
import numpy as np
from plotnine import *
import matplotlib.pyplot as plt
from io import BytesIO
import os
import threading

# ---------------- FUNÇÕES DE CÁLCULO ------------------
def calcula_umax_global_com_amplitude(df):
    Ustats = {}
    for prof in [20, 40, 60]:
        u_col = f'U{prof}'
        if u_col in df.columns:
            umax = df[u_col].max(skipna=True)
            umin = df[u_col].min(skipna=True)
            Ustats[u_col] = {
                'max': umax,
                'min': umin,
                'amplitude': umax - umin
            }
    return Ustats

def calcula_indice_microclimatico(dados, Umax_ref=None,
                                  Tref_med=25,
                                  Tref_max=35,
                                  Tref_amp=10,
                                  lim_inf_pct=0.8,
                                  lim_sup_pct=0.9,
                                  metodo_umidade="Tradicional (percentual da Umax)",
                                  aplicar_penalizacao=False,
                                  amplitude_max_global=None,
                                  alfa=0.5):
    dados = dados.copy()
    dados['Data'] = pd.to_datetime(dados['Data'], errors='coerce').dt.normalize()

    for col in dados.select_dtypes(include=[np.number]).columns:
        dados[col] = dados[col].replace(0, np.nan)

    profundidades = [20, 40, 60]
    resultados = []

    for prof in profundidades:
        u_col = f'U{prof}'
        if u_col not in dados.columns:
            continue

        umid_diaria = dados.groupby('Data')[u_col].mean()

        t_col = f'T{prof}'
        if t_col in dados.columns:
            tmax = dados.groupby('Data')[t_col].max()
            tmin = dados.groupby('Data')[t_col].min()
            tmed = dados.groupby('Data')[t_col].mean()

            resumo = pd.DataFrame({
                'Umid': umid_diaria,
                'Tmax': tmax,
                'Tmin': tmin,
                'Tmed': tmed
            }).dropna()

            resumo['Tamp'] = resumo['Tmax'] - resumo['Tmin']

            IETS = 1 - (
                (abs(resumo['Tmed'].mean() - Tref_med) / Tref_med +
                 resumo['Tmax'].mean() / Tref_max +
                 resumo['Tamp'].mean() / Tref_amp) / 3
            )
        else:
            resumo = pd.DataFrame({'Umid': umid_diaria}).dropna()
            IETS = np.nan

        if Umax_ref and u_col in Umax_ref:
            Umax_global = Umax_ref[u_col]['max']
            Umin_global = Umax_ref[u_col]['min']
            amplitude_real = Umax_ref[u_col]['amplitude']
        else:
            Umax_global = dados[u_col].max(skipna=True)
            Umin_global = dados[u_col].min(skipna=True)
            amplitude_real = Umax_global - Umin_global

        if metodo_umidade == "Baseado na amplitude real":
            lim_inf = Umax_global - (amplitude_real * lim_inf_pct)
            lim_sup = Umax_global - (amplitude_real * lim_sup_pct)
        else:
            lim_inf = Umax_global * lim_inf_pct
            lim_sup = Umax_global * lim_sup_pct

        prop = ((resumo['Umid'] >= lim_inf) & (resumo['Umid'] <= lim_sup)).mean()
        IRHE = prop

        if metodo_umidade == "Baseado na amplitude real" and aplicar_penalizacao and amplitude_max_global and amplitude_max_global > 0:
            penalizacao = 1 - alfa * (amplitude_real / amplitude_max_global)
            IRHE = IRHE * penalizacao

        IEMS = (IETS + IRHE) / 2 if not np.isnan(IETS) else IRHE

        resultado = {
            'Profundidade': prof,
            'IETS': IETS,
            'IRHE': IRHE,
            'IEMS': IEMS
        }
        if metodo_umidade == "Baseado na amplitude real":
            resultado['Amplitude_real'] = amplitude_real

        resultados.append(resultado)

    if not resultados:
        return None
    return pd.DataFrame(resultados)

def extrai_amplitude_maxima_global(planilhas):
    amplitudes = []
    for df in planilhas.values():
        uref = calcula_umax_global_com_amplitude(df)
        for estat in uref.values():
            amplitudes.append(estat['amplitude'])
    return max(amplitudes) if amplitudes else None

# ---------------- INTERFACE STREAMLIT ------------------

st.set_page_config(page_title="Índice Microclimático", layout="wide")
st.title("Calculadora de Índices Microclimáticos do Solo")

uploaded_file = st.file_uploader("Envie seu arquivo Excel com várias abas", type=["xlsx"])

# Parâmetros
st.sidebar.header("Parâmetros de referência")
Tref_med = st.sidebar.number_input("Temperatura média ideal (°C)", value=25.0)
Tref_max = st.sidebar.number_input("Temperatura máxima ideal (°C)", value=35.0)
Tref_amp = st.sidebar.number_input("Amplitude térmica ideal (°C)", value=10.0)

metodo_umidade = st.sidebar.selectbox(
    "Método para definir faixa boa de umidade:",
    ["Tradicional (percentual da Umax)", "Baseado na amplitude real"]
)

if metodo_umidade == "Baseado na amplitude real":
    lim_inf_pct = st.sidebar.slider("Faixa inferior (% da amplitude)", 0.0, 1.0, 0.6)
    lim_sup_pct = st.sidebar.slider("Faixa superior (% da amplitude)", 0.0, 1.0, 0.1)
    alfa = st.sidebar.slider("Alfa - penalização da amplitude", 0.0, 1.0, 0.5)
else:
    lim_inf_pct = st.sidebar.slider("Limite inferior umidade (% Umax)", 0.0, 1.0, 0.8)
    lim_sup_pct = st.sidebar.slider("Limite superior umidade (% Umax)", 0.0, 1.0, 0.9)
    alfa = None  # Não usado

planilhas = None
resultados = None

if uploaded_file is not None:
    xls = pd.ExcelFile(uploaded_file)
    abas = xls.sheet_names
    if not abas:
        st.error("Arquivo não possui abas válidas.")
    else:
        st.write(f"Abas encontradas: {abas}")
        planilhas = {aba: xls.parse(aba) for aba in abas}
        amplitude_max_global = extrai_amplitude_maxima_global(planilhas)

        st.write("Escolha o que deseja calcular:")

        col1, col2, col3 = st.columns(3)
        with col1:
            btn_iets = st.button("Calcular IETS")
        with col2:
            btn_irhe = st.button("Calcular IRHE")
        with col3:
            btn_iems = st.button("Calcular IEMS")

        if btn_iets or btn_irhe or btn_iems:
            resultados_lista = []
            for nome_df, df in planilhas.items():
                res = calcula_indice_microclimatico(
                    df,
                    Umax_ref=calcula_umax_global_com_amplitude(df),
                    Tref_med=Tref_med,
                    Tref_max=Tref_max,
                    Tref_amp=Tref_amp,
                    lim_inf_pct=lim_inf_pct,
                    lim_sup_pct=lim_sup_pct,
                    metodo_umidade=metodo_umidade,
                    aplicar_penalizacao=(btn_irhe or btn_iems) and metodo_umidade=="Baseado na amplitude real",
                    amplitude_max_global=amplitude_max_global,
                    alfa=alfa if alfa is not None else 0.5
                )
                if res is not None:
                    res['Ano'] = df['Data'].dt.year.min()  # Pode ajustar para ano real agrupado se quiser
                    res['Origem'] = nome_df
                    resultados_lista.append(res)

            if resultados_lista:
                resultados = pd.concat(resultados_lista, ignore_index=True)
                # Filtra as colunas conforme botão
                if btn_iets:
                    resultados = resultados[['Origem', 'Ano', 'Profundidade', 'IETS']]
                elif btn_irhe:
                    cols = ['Origem', 'Ano', 'Profundidade', 'IRHE']
                    if metodo_umidade == "Baseado na amplitude real":
                        cols.append('Amplitude_real')
                    resultados = resultados[cols]
                elif btn_iems:
                    resultados = resultados[['Origem', 'Ano', 'Profundidade', 'IETS', 'IRHE', 'IEMS']]

                st.subheader("Resultados")
                st.dataframe(resultados)

        # Botão para gerar gráfico após cálculo
        if resultados is not None:
            st.write("---")
            st.subheader("Gerar gráfico")

            indice_disponivel = [col for col in ['IETS','IRHE','IEMS'] if col in resultados.columns]
            ano_disponivel = sorted(resultados['Ano'].unique())
            profundidade_disponivel = sorted(resultados['Profundidade'].unique())

            indice_selecionado = st.selectbox("Índice", indice_disponivel)
            ano_selecionado = st.selectbox("Ano", ano_disponivel)
            profundidade_selecionada = st.selectbox("Profundidade (cm)", profundidade_disponivel)

            df_graf = resultados[
                (resultados['Ano'] == ano_selecionado) &
                (resultados['Profundidade'] == profundidade_selecionada)
            ]

            if indice_selecionado not in df_graf.columns or df_graf.empty:
                st.warning("Não há dados para o índice/ano/profundidade selecionados.")
            else:
                p = (ggplot(df_graf, aes(x='Origem', y=indice_selecionado, fill='Origem')) +
                     geom_bar(stat='identity', position='dodge') +
                     labs(title=f"{indice_selecionado} - Ano {ano_selecionado} - Profundidade {profundidade_selecionada} cm",
                          x="Talhão",
                          y=indice_selecionado) +
                     theme_minimal(base_size=14) +
                     theme(axis_text_x=element_text(angle=45, hjust=1), legend_position='none'))

                fig = p.draw()
                st.pyplot(fig, dpi=150)

                buffer = BytesIO()
                fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                buffer.seek(0)

                st.download_button(
                    label="📅 Baixar gráfico como PNG",
                    data=buffer,
                    file_name=f"grafico_{indice_selecionado}_{ano_selecionado}_{profundidade_selecionada}cm.png",
                    mime="image/png"
                )

            # Botão para baixar resultados Excel
            excel_data = BytesIO()
            with pd.ExcelWriter(excel_data, engine='openpyxl') as writer:
                resultados.to_excel(writer, index=False, sheet_name="Resultados")
            excel_data.seek(0)

            st.download_button(
                label="📄 Baixar resultados em Excel",
                data=excel_data,
                file_name="resultados_microclimaticos.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# --- Botão para encerrar o aplicativo ---
def fechar_app():
    def delayed_shutdown():
        import time
        time.sleep(1)
        os._exit(0)
    threading.Thread(target=delayed_shutdown).start()

if st.button("🚪 Encerrar aplicativo"):
    st.warning("Encerrando o aplicativo...")
    fechar_app()
