import streamlit as st
import pandas as pd
import numpy as np
from plotnine import *
import matplotlib.pyplot as plt
from io import BytesIO
import os
import threading

# ---------------- FUNÇÕES DE CÁLCULO ------------------
def calcula_umax_global(df):
    Umax = {}
    for prof in [20, 40, 60]:
        u_col = f'U{prof}'
        if u_col in df.columns:
            Umax[u_col] = df[u_col].max(skipna=True)
    return Umax

def calcula_indice_microclimatico(dados, Umax_ref=None,
                                  Tref_med=25,
                                  Tref_max=35,
                                  Tref_amp=10,
                                  lim_inf_pct=0.8,
                                  lim_sup_pct=0.9):
    dados = dados.copy()
    # Converte e normaliza a coluna Data (zera hora)
    dados['Data'] = pd.to_datetime(dados['Data'], errors='coerce').dt.normalize()

    # Substitui zeros por NaN só nas colunas numéricas
    for col in dados.select_dtypes(include=[np.number]).columns:
        dados[col] = dados[col].replace(0, np.nan)

    profundidades = [20, 40, 60]
    resultados = []

    for prof in profundidades:
        u_col = f'U{prof}'
        if u_col not in dados.columns:
            continue
        umid = dados.groupby('Data')[u_col].mean()

        t_col = f'T{prof}'
        if t_col in dados.columns:
            tmax = dados.groupby('Data')[t_col].max()
            tmin = dados.groupby('Data')[t_col].min()
            tmed = dados.groupby('Data')[t_col].mean()

            resumo = pd.DataFrame({
                'Umid': umid,
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
            resumo = pd.DataFrame({'Umid': umid}).dropna()
            IETS = np.nan

        if Umax_ref is not None and u_col in Umax_ref:
            Umax = Umax_ref[u_col]
        else:
            Umax = resumo['Umid'].max()

        lim_inf = lim_inf_pct * Umax
        lim_sup = lim_sup_pct * Umax

        prop = ((resumo['Umid'] >= lim_inf) & (resumo['Umid'] <= lim_sup)).mean()
        IRHE = prop

        if not np.isnan(IETS):
            IEMS = (IETS + IRHE) / 2
        else:
            IEMS = IRHE

        resultados.append({
            'Profundidade': prof,
            'IETS': IETS,
            'IRHE': IRHE,
            'IEMS': IEMS
        })

    if not resultados:
        return None
    return pd.DataFrame(resultados)

def calcula_por_ano_periodo(df, nome_df,
                            Tref_med=25,
                            Tref_max=35,
                            Tref_amp=10,
                            lim_inf_pct=0.8,
                            lim_sup_pct=0.9):
    df = df.copy()
    # Converte e normaliza a coluna Data (zera hora)
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
        res = calcula_indice_microclimatico(
            grupo,
            Umax_ref=Umax_ref,
            Tref_med=Tref_med,
            Tref_max=Tref_max,
            Tref_amp=Tref_amp,
            lim_inf_pct=lim_inf_pct,
            lim_sup_pct=lim_sup_pct
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

def calcula_para_varias_planilhas(nome_planilhas,
                                  Tref_med=25,
                                  Tref_max=35,
                                  Tref_amp=10,
                                  lim_inf_pct=0.8,
                                  lim_sup_pct=0.9):
    resultados = []
    for nome_df, df in nome_planilhas.items():
        res = calcula_por_ano_periodo(
            df,
            nome_df,
            Tref_med,
            Tref_max,
            Tref_amp,
            lim_inf_pct,
            lim_sup_pct
        )
        if res is not None:
            resultados.append(res)
    if not resultados:
        return None
    return pd.concat(resultados, ignore_index=True)

def gerar_png_para_download(plot, nome_arquivo="grafico.png", dpi=300):
    buffer = BytesIO()
    fig = plot.draw()
    fig.savefig(buffer, format='png', dpi=dpi, bbox_inches='tight')
    buffer.seek(0)
    return buffer

def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name="Resultados")
    return output.getvalue()

# ---------------- INTERFACE STREAMLIT ------------------
st.set_page_config(page_title="Índice Microclimático", layout="wide")
st.title("Calculadora de Índices Microclimáticos do Solo")

uploaded_file = st.file_uploader("Envie seu arquivo Excel com várias abas", type=["xlsx"])

# Parâmetros
st.sidebar.header("Parâmetros de referência")
Tref_med = st.sidebar.number_input("Temperatura média ideal (°C)", value=25.0)
Tref_max = st.sidebar.number_input("Temperatura máxima ideal (°C)", value=35.0)
Tref_amp = st.sidebar.number_input("Amplitude térmica ideal (°C)", value=10.0)
lim_inf_pct = st.sidebar.slider("Limite inferior umidade (% Umax)", 0.0, 1.0, 0.8)
lim_sup_pct = st.sidebar.slider("Limite superior umidade (% Umax)", 0.0, 1.0, 0.9)

if uploaded_file is not None:
    xls = pd.ExcelFile(uploaded_file)
    abas = xls.sheet_names
    if not abas:
        st.error("Arquivo não possui abas válidas.")
    else:
        st.write(f"Abas encontradas: {abas}")
        planilhas = {aba: xls.parse(aba) for aba in abas}

        resultados_ilp = calcula_para_varias_planilhas(
            planilhas,
            Tref_med=Tref_med,
            Tref_max=Tref_max,
            Tref_amp=Tref_amp,
            lim_inf_pct=lim_inf_pct,
            lim_sup_pct=lim_sup_pct
        )

        if resultados_ilp is not None and not resultados_ilp.empty:
            st.subheader("Resultados")
            st.dataframe(resultados_ilp)

            dados_long = resultados_ilp.melt(
                id_vars=['Ano', 'Periodo', 'Origem', 'Profundidade'],
                value_vars=['IETS', 'IRHE', 'IEMS'],
                var_name='Indice',
                value_name='Valor'
            )

            anos_disponiveis = sorted(dados_long['Ano'].unique())
            indice_disponivel = ['IETS', 'IRHE', 'IEMS']

            ano_selecionado = st.selectbox("Selecione o Ano para o gráfico", anos_disponiveis)
            indice_selecionado = st.selectbox("Selecione o Índice para o gráfico", indice_disponivel)

            df_graf = dados_long[(dados_long['Ano'] == ano_selecionado) & (dados_long['Indice'] == indice_selecionado)]

            if df_graf.empty:
                st.warning("Não há dados para o ano e índice selecionados.")
            else:
                p = (ggplot(df_graf, aes(x='Origem', y='Valor', fill='Periodo')) +
                     geom_bar(stat='identity', position='dodge') +
                     scale_fill_manual(values={'Umido':'blue', 'Seco':'red'}) +
                     labs(title=f"{indice_selecionado} - Ano {ano_selecionado}",
                          x="Talhão",
                          y=indice_selecionado,
                          fill="Período") +
                     theme_minimal(base_size=12) +
                     theme(axis_text_x=element_text(angle=45, hjust=1)))

                fig = p.draw()
                st.pyplot(fig, dpi=150)

                buffer = gerar_png_para_download(p, nome_arquivo=f"{indice_selecionado}_{ano_selecionado}.png")

                st.download_button(
                    label="📅 Baixar gráfico como PNG",
                    data=buffer,
                    file_name=f"grafico_{indice_selecionado}_{ano_selecionado}.png",
                    mime="image/png"
                )

            excel_data = to_excel(resultados_ilp)

            st.download_button(
                label="📄 Baixar resultados em Excel",
                data=excel_data,
                file_name="resultados_microclimaticos.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.warning("Não foi possível calcular os índices para os dados enviados.")
else:
    st.info("Faça upload do arquivo Excel para iniciar o cálculo.")

# --- Botão para encerrar o aplicativo ---
def fechar_app():
    def delayed_shutdown():
        import time
        time.sleep(1)
        os._exit(0)
    threading.Thread(target=delayed_shutdown).start()

st.markdown("---")
if st.button("🚪 Encerrar aplicativo"):
    st.warning("Encerrando o aplicativo...")
    fechar_app()
