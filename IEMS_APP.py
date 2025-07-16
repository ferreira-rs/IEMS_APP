import streamlit as st
import pandas as pd
import numpy as np
from plotnine import *
from io import BytesIO
import os
import threading

# ---------------- FUNÇÕES DE CÁLCULO ------------------

def calcula_umax_global(df):
    Ustats = {}
    for prof in [20, 40, 60]:
        u_col = f'U{prof}'
        if u_col in df.columns:
            Ustats[u_col] = {
                'max': df[u_col].max(skipna=True),
                'min': df[u_col].min(skipna=True)
            }
    return Ustats

def extrai_amplitude_maxima_global(planilhas):
    amplitudes = []
    for df in planilhas.values():
        Ustats = calcula_umax_global(df)
        for estat in Ustats.values():
            amplitudes.append(estat['max'] - estat['min'])
    return max(amplitudes) if amplitudes else None

def calcula_irhe(dados, Umax_ref=None,
                 lim_inf_pct=0.8,
                 lim_sup_pct=0.9,
                 metodo_umidade="Tradicional (percentual da Umax)",
                 alfa=0.5,
                 amplitude_max_global=None):
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
        resumo = pd.DataFrame({'Umid': umid_diaria}).dropna()

        if Umax_ref is not None and u_col in Umax_ref:
            Umax_global = Umax_ref[u_col]['max']
            Umin_global = Umax_ref[u_col]['min']
        else:
            Umax_global = dados[u_col].max(skipna=True)
            Umin_global = dados[u_col].min(skipna=True)

        if metodo_umidade == "Baseado na amplitude real":
            amplitude = Umax_global - Umin_global
            lim_inf = Umax_global - (amplitude * lim_inf_pct)
            lim_sup = Umax_global - (amplitude * lim_sup_pct)
        else:
            amplitude = None
            lim_inf = Umax_global * lim_inf_pct
            lim_sup = Umax_global * lim_sup_pct

        prop = ((resumo['Umid'] >= lim_inf) & (resumo['Umid'] <= lim_sup)).mean()
        IRHE = prop

        if metodo_umidade == "Baseado na amplitude real" and amplitude_max_global and amplitude_max_global > 0 and alfa > 0:
            penalizacao = 1 - alfa * (amplitude / amplitude_max_global)
            IRHE *= penalizacao

        resultados.append({
            'Profundidade': prof,
            'IRHE': IRHE
        })

    if not resultados:
        return None
    return pd.DataFrame(resultados)

def calcula_irhe_por_ano_periodo(df, nome_df,
                                 lim_inf_pct=0.8,
                                 lim_sup_pct=0.9,
                                 metodo_umidade="Tradicional (percentual da Umax)",
                                 alfa=0.5,
                                 amplitude_max_global=None):
    df = df.copy()
    df['Data'] = pd.to_datetime(df['Data'], errors='coerce').dt.normalize()
    df['Mes'] = df['Data'].dt.month
    df['Ano'] = df['Data'].dt.year
    df['Periodo'] = np.where(df['Mes'].isin([10,11,12,1,2,3]), 'Umido', 'Seco')
    df['AnoRef'] = np.where(df['Mes'].isin([1,2,3]), df['Ano'] -1, df['Ano'])

    Umax_ref = calcula_umax_global(df)

    resultados = []
    grouped = df.groupby(['AnoRef', 'Periodo'])

    for (ano_ref, periodo), grupo in grouped:
        if grupo.empty:
            continue
        res = calcula_irhe(
            grupo,
            Umax_ref=Umax_ref,
            lim_inf_pct=lim_inf_pct,
            lim_sup_pct=lim_sup_pct,
            metodo_umidade=metodo_umidade,
            alfa=alfa,
            amplitude_max_global=amplitude_max_global
        )
        if res is None:
            continue
        res['Ano'] = ano_ref
        res['Periodo'] = periodo
        res['Origem'] = nome_df
        resultados.append(res)

    if not resultados:
        return None
    return pd.concat(resultados, ignore_index=True)

def calcula_irhe_varias_planilhas(planilhas,
                                  lim_inf_pct=0.8,
                                  lim_sup_pct=0.9,
                                  metodo_umidade="Tradicional (percentual da Umax)",
                                  alfa=0.5,
                                  amplitude_max_global=None):
    resultados = []
    for nome_df, df in planilhas.items():
        res = calcula_irhe_por_ano_periodo(
            df,
            nome_df,
            lim_inf_pct=lim_inf_pct,
            lim_sup_pct=lim_sup_pct,
            metodo_umidade=metodo_umidade,
            alfa=alfa,
            amplitude_max_global=amplitude_max_global
        )
        if res is not None:
            resultados.append(res)
    if not resultados:
        return None
    return pd.concat(resultados, ignore_index=True)

def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name="Resultados")
    return output.getvalue()

# ---------------- INTERFACE STREAMLIT ------------------

st.set_page_config(page_title="Índice Microclimático", layout="wide")

# Exibe o logo
st.image("IEMS_LOGO.png", width=80)  # ajuste o width como quiser

st.title("Calculadora de Índices Microclimáticos do Solo")

uploaded_file = st.file_uploader("Envie seu arquivo Excel com várias abas", type=["xlsx"])

st.sidebar.header("Parâmetros para cálculo do IRHE")

metodo_umidade = st.sidebar.selectbox(
    "Método para definir faixa boa de umidade:",
    ["Tradicional (percentual da Umax)", "Baseado na amplitude real"]
)

if metodo_umidade == "Baseado na amplitude real":
    lim_inf_pct = st.sidebar.slider("Faixa inferior (% da amplitude)", 0.0, 1.0, 0.6)
    lim_sup_pct = st.sidebar.slider("Faixa superior (% da amplitude)", 0.0, 1.0, 0.1)
    alfa = st.sidebar.slider("Alfa (intensidade da penalização)", 0.0, 1.0, 0.5)
else:
    lim_inf_pct = st.sidebar.slider("Limite inferior umidade (% Umax)", 0.0, 1.0, 0.8)
    lim_sup_pct = st.sidebar.slider("Limite superior umidade (% Umax)", 0.0, 1.0, 0.9)
    alfa = 0.0

if uploaded_file is not None:
    xls = pd.ExcelFile(uploaded_file)
    abas = xls.sheet_names
    st.write(f"Abas encontradas: {abas}")
    planilhas = {aba: xls.parse(aba) for aba in abas}

    amplitude_max_global = None
    if metodo_umidade == "Baseado na amplitude real":
        amplitude_max_global = extrai_amplitude_maxima_global(planilhas)

    resultados_irhe = calcula_irhe_varias_planilhas(
        planilhas,
        lim_inf_pct=lim_inf_pct,
        lim_sup_pct=lim_sup_pct,
        metodo_umidade=metodo_umidade,
        alfa=alfa,
        amplitude_max_global=amplitude_max_global
    )

    if resultados_irhe is not None and not resultados_irhe.empty:
        st.subheader("Resultados do IRHE")
        st.dataframe(resultados_irhe)

        excel_data = to_excel(resultados_irhe)
        st.download_button(
            label="📄 Baixar resultados em Excel",
            data=excel_data,
            file_name="IRHE_resultados.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.warning("Não foi possível calcular o IRHE com os dados enviados.")
else:
    st.info("Faça upload do arquivo Excel para iniciar o cálculo.")

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
