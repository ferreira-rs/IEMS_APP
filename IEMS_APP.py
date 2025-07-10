def calcula_indice_microclimatico(dados, Umax_ref=None,
                                  Tref_med=25,
                                  Tref_max=35,
                                  Tref_amp=10,
                                  lim_inf_pct=0.8,
                                  lim_sup_pct=0.9):
    dados = dados.copy()
    dados['Data'] = pd.to_datetime(dados['Data'])
    # Substitui zeros por NaN só nas colunas numéricas
    for col in dados.select_dtypes(include=[np.number]).columns:
        dados[col] = dados[col].replace(0, np.nan)

    profundidades = [20, 40, 60]
    resultados = []

    for prof in profundidades:
        u_col = f'U{prof}'
        t_col = f'T{prof}'

        # Se não tiver coluna de umidade, ignora esta profundidade
        if u_col not in dados.columns:
            continue

        umid = dados.groupby('Data')[u_col].mean()

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
            IETS = np.nan  # Se não tem temperatura, deixa NaN para IETS

        if Umax_ref is not None and u_col in Umax_ref:
            Umax = Umax_ref[u_col]
        else:
            Umax = resumo['Umid'].max()

        lim_inf = lim_inf_pct * Umax
        lim_sup = lim_sup_pct * Umax

        prop = ((resumo['Umid'] >= lim_inf) & (resumo['Umid'] <= lim_sup)).mean()
        IRHE = prop

        # Se IETS é NaN (faltando temp), calcula IEMS só com IRHE
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
