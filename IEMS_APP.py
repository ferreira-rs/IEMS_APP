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
        if metodo_umidade == "Baseado na amplitude real" and aplicar_penalizacao:
            resultado['Amplitude_real'] = amplitude_real

        resultados.append(resultado)

    if not resultados:
        return None
    return pd.DataFrame(resultados)

def calcula_por_ano_periodo(df, nome_df,
                            Tref_med=25,
                            Tref_max=35,
                            Tref_amp=10,
                            lim_inf_pct=0.8,
                            lim_sup_pct=0.9,
                            metodo_umidade="Tradicional (percentual da Umax)",
                            aplicar_penalizacao=False,
                            amplitude_max_global=None,
                            alfa=0.5):
    df = df.copy()
    df['Data'] = pd.to_datetime(df['Data'], errors='coerce').dt.normalize()
    df['Mes'] = df['Data'].dt.month
    df['Ano'] = df['Data'].dt.year
    df['Periodo'] = np.where(df['Mes'].isin([10,11,12,1,2,3]), 'Umido', 'Seco')
    df['AnoRef'] = np.where(df['Mes'].isin([1,2,3]), df['Ano'] -1, df['Ano'])

    Umax_ref = calcula_umax_global_com_amplitude(df)

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
            lim_sup_pct=lim_sup_pct,
            metodo_umidade=metodo_umidade,
            aplicar_penalizacao=aplicar_penalizacao,
            amplitude_max_global=amplitude_max_global,
            alfa=alfa
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
                                  lim_sup_pct=0.9,
                                  metodo_umidade="Tradicional (percentual da Umax)",
                                  aplicar_penalizacao=False,
                                  amplitude_max_global=None,
                                  alfa=0.5):
    resultados = []
    for nome_df, df in nome_planilhas.items():
        res = calcula_por_ano_periodo(
            df,
            nome_df,
            Tref_med,
            Tref_max,
            Tref_amp,
            lim_inf_pct,
            lim_sup_pct,
            metodo_umidade,
            aplicar_penalizacao,
            amplitude_max_global,
            alfa
        )
        if res is not None:
            resultados.append(res)
    if not resultados:
        return None
    return pd.concat(resultados, ignore_index=True)

def extrai_amplitude_maxima_global(planilhas):
    amplitudes = []
    for df in planilhas.values():
        uref = calcula_umax_global_com_amplitude(df)
        for estat in uref.values():
            amplitudes.append(estat['amplitude'])
    return max(amplitudes) if amplitudes else None

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

# Exibe o logo
st.image("IEMS_LOGO.png", width=150)  # ajuste o width como quiser

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
    aplicar_penalizacao = st.sidebar.checkbox("Aplicar penalização por amplitude")
    alfa = st.sidebar.slider("Alfa (intensidade da penalização)", 0.0, 1.0, 0.5)
else:
    lim_inf_pct = st.sidebar.slider("Limite inferior umidade (% Umax)", 0.0, 1.0, 0.8)
    lim_sup_pct = st.sidebar.slider("Limite superior umidade (% Umax)", 0.0, 1.0, 0.9)
    aplicar_penalizacao = False
    alfa = 0.0

# Botões de cálculo
st.subheader("Escolha a ação")
calcular_iets = st.button("Calcular apenas IETS")
calcular_irhe = st.button("Calcular apenas IRHE")
calcular_iems = st.button("Calcular IEMS completo")
gerar_grafico = st.button("📊 Gerar gráfico")

planilhas = {}
resultados = None
amplitude_max_global = None

if uploaded_file is not None:
    xls = pd.ExcelFile(uploaded_file)
    abas = xls.sheet_names
    if abas:
        st.success(f"{len(abas)} abas carregadas: {abas}")
        planilhas = {aba: xls.parse(aba) for aba in abas}
        if metodo_umidade == "Baseado na amplitude real" and aplicar_penalizacao:
            amplitude_max_global = extrai_amplitude_maxima_global(planilhas)

        if calcular_iets:
            resultados = calcula_para_varias_planilhas(
                planilhas, Tref_med, Tref_max, Tref_amp,
                lim_inf_pct, lim_sup_pct,
                metodo_umidade, False, None, 0.0
            )
            if resultados is not None:
                st.subheader("Resultado - IETS")
                st.dataframe(resultados[['Ano', 'Periodo', 'Origem', 'Profundidade', 'IETS']])
        elif calcular_irhe:
            resultados = calcula_para_varias_planilhas(
                planilhas, Tref_med, Tref_max, Tref_amp,
                lim_inf_pct, lim_sup_pct,
                metodo_umidade, aplicar_penalizacao, amplitude_max_global, alfa
            )
            if resultados is not None:
                st.subheader("Resultado - IRHE (com penalização e amplitude)")
                cols = ['Ano', 'Periodo', 'Origem', 'Profundidade', 'IRHE']
                if aplicar_penalizacao and metodo_umidade == "Baseado na amplitude real":
                    cols.append('Amplitude_real')
                st.dataframe(resultados[cols])
        elif calcular_iems:
            resultados = calcula_para_varias_planilhas(
                planilhas, Tref_med, Tref_max, Tref_amp,
                lim_inf_pct, lim_sup_pct,
                metodo_umidade, aplicar_penalizacao, amplitude_max_global, alfa
            )
            if resultados is not None:
                st.subheader("Resultado - IEMS completo")
                st.dataframe(resultados[['Ano', 'Periodo', 'Origem', 'Profundidade', 'IETS', 'IRHE', 'IEMS']])

        if resultados is not None and gerar_grafico:
            dados_long = resultados.melt(
                id_vars=['Ano', 'Periodo', 'Origem', 'Profundidade'],
                value_vars=['IETS', 'IRHE', 'IEMS'],
                var_name='Indice',
                value_name='Valor'
            )

            anos_disponiveis = sorted(dados_long['Ano'].unique())
            indices_disponiveis = ['IETS', 'IRHE', 'IEMS']

            ano_selecionado = st.selectbox("Selecione o Ano para o gráfico", anos_disponiveis)
            indice_selecionado = st.selectbox("Selecione o Índice para o gráfico", indices_disponiveis)

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

            excel_data = to_excel(resultados)

            st.download_button(
                label="📄 Baixar resultados em Excel",
                data=excel_data,
                file_name="resultados_microclimaticos.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.error("Arquivo não possui abas válidas.")
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
