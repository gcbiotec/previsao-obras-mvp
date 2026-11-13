"""Piloto: previsão de término de obras (AFT Log-Normal) em Streamlit."""
from __future__ import annotations

import io
import os
from datetime import date
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import aft_core as core

MARCOS = core.MARCOS_DIAS  # (0, 30, 60): dias depois do prazo original

# Para pré-carregar um CSV sem usar o upload (útil rodando localmente):
#   PREVISAO_CSV=caminho/contratos.csv streamlit run app.py
CSV_PRE_CARREGADO = os.environ.get("PREVISAO_CSV", "").strip()

# Cores (checadas com o validador da skill dataviz; funcionam em tema claro e escuro).
COR_CURVA = "#3987e5"                                         # série única: azul
RAMPA_FAIXAS = ["#86b6ef", "#3987e5", "#256abf", "#184f95"]   # ordinal: claro -> escuro = mais tarde
TEXTO_FAIXAS = ["#0b0b0b", "#ffffff", "#ffffff", "#ffffff"]   # rótulo dentro de cada bloco
COR_REFERENCIA = "#898781"                                    # linhas de referência e rótulos (neutro)
COR_DESTAQUE = "#eb6834"                                      # o resultado real

st.set_page_config(page_title="Previsão de término de obras", page_icon="🏗️", layout="wide")

# Garante que o app e o núcleo são da mesma versão (evita um KeyError difícil de entender após um deploy parcial).
if not hasattr(core, "MARCOS_LONGOS_DIAS"):
    st.error(
        "O arquivo **aft_core.py** carregado está desatualizado em relação ao app.py. "
        f"Arquivo em uso: `{core.__file__}`. Envie o aft_core.py novo ao repositório (junto com o app.py) "
        "e depois use *Manage app → Reboot app*."
    )
    st.stop()

# Datas e rótulos longos cortavam nos cartões (st.metric); reduz a fonte do valor e deixa o rótulo quebrar linha.
st.markdown(
    """
<style>
[data-testid="stMetricValue"] { font-size: 1.55rem; }
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
    white-space: normal !important; overflow: visible !important; text-overflow: clip !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Dados e modelo (em cache: o ajuste leva milissegundos, mas evita refazer a cada clique)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def ler_csv(conteudo: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(conteudo), encoding="utf-8-sig", sep=None, engine="python")


@st.cache_data(show_spinner=False)
def ajustar(contratos: pd.DataFrame, data_corte: date, alpha: float):
    df_prep = core.preparar_contratos(contratos, data_corte)
    modelo = core.treinar_modelo(df_prep, alpha)
    return df_prep, modelo


@st.cache_data(show_spinner=False)
def modelo_da_epoca(df_prep: pd.DataFrame, cutoff: date, alpha: float):
    return core.treinar_no_passado(df_prep, cutoff, alpha)


# --------------------------------------------------------------------------- #
# Formatação
# --------------------------------------------------------------------------- #
def fmt_pct(x: float) -> str:
    if x <= 0:
        return "0%"
    if x >= 1:
        return "100%"
    if x < 0.005:
        return "<1%"
    if x > 0.995:
        return ">99%"
    return f"{100 * x:.0f}%"


def fmt_data(d) -> str:
    return pd.Timestamp(d).strftime("%d/%m/%Y")


def fmt_dias(n: float) -> str:
    return f"{n:,.0f}".replace(",", ".") + " dias"


def soma_dias(data_os, dias: float) -> pd.Timestamp:
    return pd.Timestamp(data_os) + pd.Timedelta(days=int(round(dias)))


def n_dias(n: float) -> str:
    n = int(round(abs(n)))
    return f"{n} dia" if n == 1 else f"{n} dias"


def titulo_marco(offset: int) -> str:
    return "Até o prazo original" if offset == 0 else f"Até {offset} dias depois"


# --------------------------------------------------------------------------- #
# Gráficos
# --------------------------------------------------------------------------- #
def grafico_faixas(faixas: list):
    """Barra segmentada: em que faixa a obra deve terminar (as partes somam 100%)."""
    nomes = [n for n, _ in faixas]
    n = len(nomes)
    idx = [round(i * (len(RAMPA_FAIXAS) - 1) / max(n - 1, 1)) for i in range(n)]
    cores = [RAMPA_FAIXAS[i] for i in idx]
    textos = [TEXTO_FAIXAS[i] for i in idx]

    x1 = np.cumsum([p for _, p in faixas])
    x0 = x1 - np.array([p for _, p in faixas])
    df = pd.DataFrame({
        "faixa": nomes,
        "prob": [p for _, p in faixas],
        "x0": x0,
        "x1": x1,
        "linha": "total",
        "rotulo": [fmt_pct(p) if p >= 0.07 else "" for _, p in faixas],  # bloco estreito não leva rótulo
        "cor_texto": textos,
        "prob_txt": [fmt_pct(p) for _, p in faixas],
    })

    barras = alt.Chart(df).mark_bar(size=28, cornerRadius=4).encode(
        x=alt.X("x0:Q", title=None, scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", tickCount=5)),
        x2="x1:Q",
        y=alt.Y("linha:N", axis=None),
        color=alt.Color(
            "faixa:N", sort=nomes,
            scale=alt.Scale(domain=nomes, range=cores),
            legend=alt.Legend(orient="bottom", title=None, symbolType="square"),
        ),
        tooltip=[alt.Tooltip("faixa:N", title="Faixa"), alt.Tooltip("prob_txt:N", title="Probabilidade")],
    )
    rotulos = alt.Chart(df[df["rotulo"] != ""]).mark_text(
        align="left", dx=8, fontSize=12, fontWeight="bold"
    ).encode(
        x="x0:Q", y=alt.Y("linha:N", axis=None), text="rotulo:N",
        color=alt.Color("cor_texto:N", scale=None, legend=None),
    )
    return (barras + rotulos).properties(height=110)


def grafico_curva(modelo, data_os, prazo, vigente, decorridos, p50_dias, p90_dias, real_dias=None):
    """Curva: probabilidade de a obra já ter terminado em cada data, com marcos de referência."""
    ja_rel = decorridos / prazo
    x_max = max(p90_dias * 1.1, vigente * 1.25, decorridos * 1.1, prazo * 1.5)
    if real_dias:
        x_max = max(x_max, real_dias * 1.1)
    dias = np.linspace(max(decorridos, 1), x_max, 160)
    cdf = modelo.cdf_condicional(prazo, ja_rel, dias / prazo)
    curva = pd.DataFrame({
        "data": pd.Timestamp(data_os) + pd.to_timedelta(dias, unit="D"),
        "prob": cdf,
    })

    marcas = [("Prazo original", prazo, [6, 4])]
    if vigente > prazo:
        marcas.append(("Prazo aditivado", vigente, [2, 3]))
    if decorridos > 0:
        marcas.append(("Hoje", decorridos, [1, 0]))
    marcas.append(("P50", p50_dias, [8, 2, 2, 2]))
    if real_dias:
        marcas.append(("Término real", real_dias, [1, 0]))

    rot = [m[0] for m in marcas]
    df_m = pd.DataFrame({
        "rotulo": rot,
        "data": [pd.Timestamp(data_os) + pd.Timedelta(days=float(m[1])) for m in marcas],
        "y": [max(0.96 - 0.085 * i, 0.05) for i in range(len(marcas))],
    })
    meio = curva["data"].iloc[len(curva) // 2]
    df_m["lado"] = np.where(df_m["data"] > meio, "dir", "esq")

    cores = [COR_DESTAQUE if r == "Término real" else COR_REFERENCIA for r in rot]
    tracos = [m[2] for m in marcas]

    x_enc = alt.X("data:T", title=None, axis=alt.Axis(format="%d/%m/%y", tickCount=6, labelOverlap=True))
    y_enc = alt.Y("prob:Q", title="Prob. de já ter terminado", axis=alt.Axis(format="%"),
                  scale=alt.Scale(domain=[0, 1]))

    linha = alt.Chart(curva).mark_line(strokeWidth=2, color=COR_CURVA).encode(x=x_enc, y=y_enc)
    pontos = alt.Chart(curva).mark_point(opacity=0, size=140).encode(
        x="data:T", y="prob:Q",
        tooltip=[alt.Tooltip("data:T", title="Data", format="%d/%m/%Y"),
                 alt.Tooltip("prob:Q", title="Prob. de já ter terminado", format=".0%")],
    )
    regras = alt.Chart(df_m).mark_rule(strokeWidth=1.5).encode(
        x="data:T",
        color=alt.Color("rotulo:N", scale=alt.Scale(domain=rot, range=cores), legend=None),
        strokeDash=alt.StrokeDash("rotulo:N", scale=alt.Scale(domain=rot, range=tracos), legend=None),
        tooltip=[alt.Tooltip("rotulo:N", title=""), alt.Tooltip("data:T", title="Data", format="%d/%m/%Y")],
    )
    camadas = [linha, pontos, regras]
    for lado, align, dx in (("esq", "left", 5), ("dir", "right", -5)):
        sub = df_m[df_m["lado"] == lado]
        if len(sub):
            camadas.append(
                alt.Chart(sub).mark_text(align=align, baseline="middle", dx=dx, fontSize=11, color=COR_REFERENCIA)
                .encode(x="data:T", y=alt.Y("y:Q", title="Prob. de já ter terminado", axis=alt.Axis(format="%"),
                                           scale=alt.Scale(domain=[0, 1])), text="rotulo:N")
            )
    return alt.layer(*camadas).properties(height=320)


# --------------------------------------------------------------------------- #
# Painel de previsão (usado no simulador e na aba de contratos reais)
# --------------------------------------------------------------------------- #
def titulo_marco_adit(offset: int) -> str:
    return "Até o prazo aditivado" if offset == 0 else f"Até {offset} dias depois do prazo aditivado"


def linha_marcos(itens, data_os, decorridos, titulo):
    """Uma linha de cartões com a probabilidade acumulada em cada marco."""
    cols = st.columns(len(itens))
    for col, it in zip(cols, itens):
        col.metric(
            titulo(it["offset"]),
            fmt_pct(it["prob_ate"]),
            help=("Probabilidade acumulada: a obra concluir **até** essa data (no dia ou antes), "
                  "dado que ainda não terminou hoje."),
        )
        passou = decorridos >= it["limite_dias"]
        col.caption(fmt_data(soma_dias(data_os, it["limite_dias"])) + ("  ·  já passou" if passou else ""))


def painel_previsao(modelo, data_os, prazo, vigente, decorridos, real_dias=None):
    """Mostra marcos, faixas, P50/P80 e curva. `vigente=None` = sem aditivo conhecido. Devolve o dict da previsão."""
    prazo = float(prazo)
    vigente_dias = float(vigente) if vigente else prazo
    p = core.previsao_contrato(modelo, prazo, decorridos, vigente_dias, MARCOS)
    adit = int(round(vigente_dias - prazo)) if p["vigente"] else 0

    st.markdown("##### Probabilidade de a obra terminar…")
    if p["vigente"]:
        st.caption("**Contados do prazo original**")
    linha_marcos(p["original"], data_os, decorridos, titulo_marco)
    if p["vigente"]:
        st.caption(f"**Contados do prazo aditivado** (prazo original + {adit} dias de aditivo)")
        linha_marcos(p["vigente"], data_os, decorridos, titulo_marco_adit)

    ref = [("Prazo original", p["original"] + p["original_longo"])]
    if p["vigente"]:
        ref.append(("Prazo aditivado", p["vigente"] + p["vigente_longo"]))
    tabela = pd.DataFrame([
        {
            "Referência": nome,
            "Marco": "No prazo" if it["offset"] == 0 else f"+{it['offset']} dias",
            "Data limite": soma_dias(data_os, it["limite_dias"]).date(),
            "Prob. de terminar até a data": it["prob_ate"],
        }
        for nome, itens in ref for it in itens
    ])
    with st.expander("Ver todos os marcos em tabela (inclui 120, 150 e 180 dias)"
                     + (" e o prazo aditivado" if p["vigente"] else ""),
                     expanded=bool(p["vigente"])):
        st.dataframe(
            tabela, hide_index=True, width="stretch",
            column_config={"Prob. de terminar até a data": st.column_config.ProgressColumn(
                min_value=0, max_value=1, format="percent")},
        )

    st.markdown("##### Em que faixa a obra deve terminar?")
    st.caption("As faixas são excludentes e somam 100%. Faixas mais escuras = término mais tardio. "
               "Cada régua conta os dias depois do prazo indicado.")
    blocos = [("Prazo original", p["faixas"], p["faixas_longas"])]
    if p["vigente"]:
        blocos.append(("Prazo aditivado", p["faixas_vigente"], p["faixas_longas_vigente"]))
    for nome, curtas, longas in blocos:
        st.markdown(f"**{nome}**")
        st.caption("Faixas curtas: até o prazo, 30 e 60 dias depois")
        st.altair_chart(grafico_faixas(curtas))
        st.caption("Faixas longas: até 120, 150 e 180 dias depois, e mais que isso")
        st.altair_chart(grafico_faixas(longas))

    q = p["quantis"]
    st.markdown("##### Quando deve terminar")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Duração P50 (mediana)", fmt_dias(q["p50"]), help="50% de chance de a obra terminar até aqui (dias contados da OS).")
    m1.caption(f"{q['p50'] / prazo:.2f}× o prazo original")
    m2.metric("Término P50 (mediana)", fmt_data(soma_dias(data_os, q["p50"])))
    m3.metric("Término P80 (pessimista)", fmt_data(soma_dias(data_os, q["p80"])), help="80% de chance de terminar até esta data.")
    m4.metric(
        "Prob. de passar do prazo aditivado" if p["vigente"] else "Prob. de passar do prazo",
        fmt_pct(p["prob_estourar_vigente"]),
    )
    st.caption(
        f"Faixa central de 80%: entre **{fmt_data(soma_dias(data_os, q['p10']))}** e "
        f"**{fmt_data(soma_dias(data_os, q['p90']))}** ({q['p10']:.0f} a {q['p90']:.0f} dias após a OS). "
        f"Tudo é condicional ao fato de a obra *ainda não ter terminado* no dia {decorridos:.0f}."
    )

    st.altair_chart(grafico_curva(modelo, data_os, prazo, vigente_dias, decorridos, q["p50"], q["p90"], real_dias))
    return p


# --------------------------------------------------------------------------- #
# Barra lateral
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Dados")
    arquivo = st.file_uploader(
        "CSV de contratos", type="csv",
        help="Os dados enviados ficam só na memória da sessão; nada é gravado no repositório.",
    )
    if arquivo is not None:
        contratos = ler_csv(arquivo.getvalue())
    elif CSV_PRE_CARREGADO and Path(CSV_PRE_CARREGADO).is_file():
        contratos = ler_csv(Path(CSV_PRE_CARREGADO).read_bytes())
        st.caption(f"CSV pré-carregado: {Path(CSV_PRE_CARREGADO).name}")
    else:
        st.info("Envie o CSV de contratos para começar.")
        st.stop()

    data_corte = st.date_input(
        "Data de corte (\"hoje\")",
        value=date.today(),
        format="DD/MM/YYYY",
        help="Contratos sem conclusão até esta data são tratados como censurados (ainda em andamento).",
    )

    if {"data_fim_observacao", "evento_conclusao"} <= set(contratos.columns):
        # a base já traz a data de corte com que foi gerada (a dos contratos em andamento)
        _cortes = pd.to_datetime(
            contratos.loc[contratos["evento_conclusao"].eq(0), "data_fim_observacao"], errors="coerce"
        ).dropna().unique()
        if len(_cortes) == 1 and pd.Timestamp(_cortes[0]).date() != data_corte:
            st.caption(f"Esta base foi gerada com corte em {fmt_data(_cortes[0])}.")

    with st.expander("Avançado"):
        alpha = st.select_slider(
            "Regularização (alpha)", options=core.ALPHAS, value=core.ALPHA_PADRAO
        )

faltantes = core.colunas_faltantes(contratos)
if faltantes:
    st.error("O CSV não tem as colunas obrigatórias: " + ", ".join(faltantes))
    st.stop()

df_prep, modelo = ajustar(contratos, data_corte, alpha)
em_andamento = core.prever_em_andamento(df_prep, modelo)

n_invalidos = int((~df_prep["flag_apto_modelagem"].astype(bool)).sum()) - int(
    df_prep["grupo_modelagem"].eq("SUSPENSO_PARALISADO").sum()
)

# --------------------------------------------------------------------------- #
# Cabeçalho
# --------------------------------------------------------------------------- #
st.title("Previsão de término de obras")

if n_invalidos > 0:
    st.caption(f"⚠️ {n_invalidos} linha(s) sem data de OS ou prazo válido foram ignoradas.")
if not modelo.convergiu:
    st.warning("O otimizador não convergiu. Revise os dados antes de usar os resultados.")

aba_sim, aba_real, aba_and, aba_sobre = st.tabs(
    ["🔮 Simulador", "🔎 Contratos reais", "📋 Contratos em andamento", "ℹ️ Sobre"]
)

# --------------------------------------------------------------------------- #
# Aba 1 — Simulador
# --------------------------------------------------------------------------- #
with aba_sim:
    st.subheader("Simule uma obra")
    with st.expander("Como preencher"):
        st.markdown(
            """
- **Prazo original:** o prazo da ordem de serviço, antes de qualquer aditivo.
- **Aditivos de prazo (dias):** o total de dias acrescentados por aditivos. *Exemplo:* prazo original de 360 dias
  e 90 dias de aditivo (60 + 30) = **prazo aditivado de 450 dias**. Deixe 0 se não houve aditivo.
- **Hoje:** o modelo calcula quantos dias já se passaram desde a OS. Se a obra ainda não começou, deixe a data de hoje
  igual à data da OS.
- A previsão assume que a obra **ainda não terminou** hoje.
"""
        )

    esq, dir_ = st.columns([1, 2], gap="large")

    with esq:
        st.markdown("##### 1. Contrato")
        data_os_sim = st.date_input(
            "Data da OS (início da obra)", value=data_corte, key="sim_os", format="DD/MM/YYYY",
        )
        prazo = st.number_input(
            "Prazo original (dias)", 30, 3650, 360, step=30, key="sim_prazo",
            help="Prazo da ordem de serviço, antes de qualquer aditivo.",
        )

        total_adit = int(st.number_input(
            "Aditivos de prazo (dias)", 0, 3650, 0, step=30, key="sim_adit_dias",
            help="Total de dias acrescentados por aditivos. Deixe 0 se não houve aditivo.",
        ))
        vigente = int(prazo) + total_adit

        st.markdown("##### 2. Situação hoje")
        hoje_sim = st.date_input("Data de hoje (referência)", value=data_corte, key="sim_hoje", format="DD/MM/YYYY")
        decorridos = max((hoje_sim - data_os_sim).days, 0)
        if hoje_sim < data_os_sim:
            st.warning("A data de hoje é anterior à OS; considerando obra ainda não iniciada (0 dias).")

    with dir_:
        st.markdown("##### Resumo do contrato")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Término original", fmt_data(soma_dias(data_os_sim, prazo)), help="Data da OS + prazo original.")
        r1.caption(f"{int(prazo)} dias de prazo")
        r2.metric("Aditivos de prazo", f"+{total_adit} dias" if total_adit else "Nenhum")
        r3.metric("Término aditivado", fmt_data(soma_dias(data_os_sim, vigente)), help="Término original + aditivos.")
        r3.caption(f"{vigente} dias de prazo")
        r4.metric("Hoje", f"dia {decorridos}", help="Dias desde a OS até a data de referência.")
        r4.caption(f"{100 * decorridos / prazo:.0f}% do prazo original")

        if decorridos >= vigente:
            st.warning(
                "Hoje já passou do prazo aditivado e a obra não terminou: a probabilidade de estourar é de 100%."
                if total_adit else
                "Hoje já passou do prazo original e a obra não terminou: a probabilidade de estourar é de 100%."
            )
        elif decorridos >= prazo:
            st.info("Hoje já passou do prazo original, mas ainda está dentro do prazo aditivado.")

        painel_previsao(modelo, data_os_sim, prazo, vigente if total_adit else None, decorridos)

# --------------------------------------------------------------------------- #
# Aba 2 — Contratos reais (seleção e verificação)
# --------------------------------------------------------------------------- #
with aba_real:
    st.subheader("Verifique a previsão em contratos reais")
    st.caption(
        "Escolha um contrato da base. Se ele ainda está em andamento, veja a previsão de hoje. "
        "Se já terminou, volte no tempo: o app refaz a previsão num ponto da obra e compara com o que aconteceu."
    )

    reais = df_prep[df_prep["data_os"].notna() & (df_prep["prazo_original_dias"] > 0)].copy()
    filtro = st.radio("Mostrar", ["Todos", "Em andamento", "Concluídos"], horizontal=True, key="real_filtro")
    if filtro == "Em andamento":
        reais = reais[reais["grupo_modelagem"] != "CONCLUIDO_COM_TARGET"]
    elif filtro == "Concluídos":
        reais = reais[reais["grupo_modelagem"] == "CONCLUIDO_COM_TARGET"]
    reais = reais.sort_values("id_contrato")

    SITUACAO = {
        "CONCLUIDO_COM_TARGET": "concluído",
        "EM_ANDAMENTO_CENSURADO": "em andamento",
        "SUSPENSO_PARALISADO": "suspenso/paralisado",
    }

    def _txt(r, col):
        v = r.get(col)
        return "" if v is None or pd.isna(v) else str(v)

    def rotulo_contrato(r) -> str:
        partes = [str(r["id_contrato"])]
        for col in ("Numero_contrato", "Nome_predio", "classificacao_contrato"):
            if _txt(r, col):
                partes.append(_txt(r, col))
        partes += [f"OS {fmt_data(r['data_os'])}", SITUACAO[r["grupo_modelagem"]]]
        return " · ".join(partes)

    if reais.empty:
        st.info("Nenhum contrato neste filtro.")
    else:
        rotulos = {r["id_contrato"]: rotulo_contrato(r) for _, r in reais.iterrows()}
        ids = list(rotulos)
        concl_recentes = reais[reais["grupo_modelagem"] == "CONCLUIDO_COM_TARGET"].sort_values("data_os")
        idx_padrao = ids.index(concl_recentes.iloc[-1]["id_contrato"]) if len(concl_recentes) else 0
        escolhido = st.selectbox(
            "Contrato", ids, index=idx_padrao, format_func=lambda i: rotulos[i], key="real_contrato"
        )
        c = reais[reais["id_contrato"] == escolhido].iloc[0]

        data_os = c["data_os"]
        prazo_c = int(c["prazo_original_dias"])
        vigente_c = int(round(c["prazo_vigente_atual_dias"]))
        adit_c = int(round(c["dias_aditivados_app"]))
        da_tabela = bool(c["aditivos_da_tabela"])
        n_adit_c = _txt(c, "n_aditivos_prazo")
        grupo = c["grupo_modelagem"]
        concluido = grupo == "CONCLUIDO_COM_TARGET"
        status_txt = str(c.get("status_final_controle", "—"))

        st.markdown("##### Dados do contrato")
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Ordem de serviço", fmt_data(data_os))
        a1.caption(f"Status no controle: {status_txt}")
        if _txt(c, "Nome_predio") or _txt(c, "Numero_contrato"):
            st.markdown(
                f"**Prédio:** {_txt(c, 'Nome_predio') or '—'}  ·  **Contrato nº** {_txt(c, 'Numero_contrato') or '—'}"
                + (f"  ·  **Tipo:** {_txt(c, 'classificacao_contrato')}" if _txt(c, "classificacao_contrato") else "")
            )
        a2.metric("Prazo original", fmt_dias(prazo_c))
        a2.caption(f"Término original: {fmt_data(soma_dias(data_os, prazo_c))}")
        a3.metric("Aditivos de prazo", f"+{adit_c} dias" if adit_c else "Nenhum")
        a3.caption(
            (f"{int(float(n_adit_c))} aditivo(s) na tabela de aditivos" if n_adit_c else "Soma dos aditivos de prazo")
            if da_tabela else "Término vigente − término original"
        )
        a4.metric("Prazo vigente", fmt_dias(vigente_c))
        a4.caption(f"Término vigente: {fmt_data(soma_dias(data_os, vigente_c))}")
        if da_tabela:
            if int(c.get("flag_dias_divergem_por_data", 0) or 0):
                por_data = c.get("dias_aditivados_por_data")
                st.warning(
                    f"A soma dos aditivos ({adit_c} dias) difere da diferença entre as datas"
                    + (f" ({int(por_data)} dias)" if por_data is not None and pd.notna(por_data) else "")
                    + ". O app usa a soma da tabela de aditivos."
                )
            if int(c.get("flag_aditivo_pos_observacao", 0) or 0):
                st.info(
                    "Parte do aditivo foi assinada depois da conclusão da obra, então não era conhecida "
                    "durante a execução."
                )
        elif adit_c:
            st.caption("Este CSV traz só as datas: os dias aditivados são calculados pela diferença entre elas.")

        if concluido:
            conclusao = c["data_conclusao_definitiva"]
            dur = int(c["tempo_observado_dias"])
            b1, b2, b3 = st.columns(3)
            b1.metric("Conclusão registrada", fmt_data(conclusao))
            b2.metric("Duração real", fmt_dias(dur))
            b2.caption(f"{dur / prazo_c:.2f}× o prazo original")
            if status_txt.upper().startswith("EM ANDAMENTO") or int(c.get("flag_status_desatualizado", 0) or 0):
                st.warning(
                    "No controle este contrato consta como EM ANDAMENTO, mas tem data de conclusão preenchida. "
                    "O modelo o trata como concluído (regra do notebook)."
                )
            if conclusao > pd.Timestamp(data_corte):
                st.warning("A data de conclusão registrada é posterior à data de corte.")

            st.divider()
            st.markdown("##### Volte no tempo e confira a previsão")
            dia = (
                st.slider("Dia da obra em que a previsão é feita (0 = dia da OS)", 0, dur - 1, 0, key=f"real_dia_{escolhido}")
                if dur > 1 else 0
            )
            data_previsao = pd.Timestamp(data_os) + pd.Timedelta(days=dia)
            tipo_modelo = st.radio(
                "Modelo usado na previsão",
                ["Modelo da época (só o que se sabia naquele dia)", "Modelo atual (já viu o desfecho desta obra)"],
                key="real_modelo",
                help="O modelo da época é a verificação honesta: não conhece contratos concluídos depois daquela data. "
                     "O modelo atual já foi treinado com este contrato e tende a parecer melhor do que realmente seria.",
            )
            m_uso = modelo
            if tipo_modelo.startswith("Modelo da época"):
                m_ep, info = modelo_da_epoca(df_prep, data_previsao.date(), alpha)
                if m_ep is None:
                    st.warning(
                        f"Em {fmt_data(data_previsao)} só havia {info['eventos']} obra(s) concluída(s) "
                        f"(mínimo de {core.MIN_EVENTOS_PASSADO} para ajustar o modelo). Usando o modelo atual, "
                        "que é otimista para este contrato."
                    )
                else:
                    m_uso = m_ep
                    st.caption(
                        f"Modelo ajustado só com o que se sabia em {fmt_data(data_previsao)}: "
                        f"{info['eventos']} obras concluídas e {info['censurados']} em andamento."
                    )
            else:
                st.caption("⚠️ Este contrato faz parte do treino do modelo atual: a comparação é otimista.")

            p = painel_previsao(m_uso, data_os, prazo_c, None, dia, real_dias=dur)

            # --- Previsão × realidade ------------------------------------------------
            st.divider()
            st.markdown("##### Previsão × realidade")
            q = p["quantis"]
            atraso = (pd.Timestamp(conclusao) - soma_dias(data_os, prazo_c)).days
            i_faixa = core.faixa_do_atraso(atraso, MARCOS)
            nome_faixa, prob_faixa = p["faixas"][i_faixa]
            prob_real = float(core.prob_ate(m_uso, prazo_c, dia, dur)[0])
            dentro = q["p10"] <= dur <= q["p90"]

            v1, v2, v3, v4 = st.columns(4)
            v1.metric("Término real", fmt_data(conclusao))
            v1.caption(f"{dur} dias após a OS")
            v2.metric("Previsão P50", fmt_data(soma_dias(data_os, q["p50"])))
            v2.caption(f"a obra terminou {n_dias(dur - q['p50'])} {'depois' if dur > q['p50'] else 'antes'} do P50")
            v3.metric("Dentro da faixa de 80%?", "Sim" if dentro else "Não",
                      help="Faixa central entre P10 e P90 da previsão.")
            v3.caption(f"{fmt_data(soma_dias(data_os, q['p10']))} a {fmt_data(soma_dias(data_os, q['p90']))}")
            v4.metric("Chance dada à data real", fmt_pct(prob_real),
                      help="Probabilidade que o modelo dava de a obra terminar até a data em que de fato terminou. "
                           "Perto de 50% = acertou o centro; perto de 0% ou 100% = errou para um lado.")
            v4.caption("de terminar até essa data")

            posicao = (f"{n_dias(atraso)} {'depois' if atraso > 0 else 'antes'} do prazo original"
                       if atraso != 0 else "exatamente no prazo original")
            st.markdown(
                f"A obra terminou **{posicao}** → faixa **{nome_faixa}**. "
                f"O modelo dava **{fmt_pct(prob_faixa)}** para essa faixa."
            )
            if adit_c:
                st.caption(
                    f"Este contrato recebeu +{adit_c} dias de aditivo ao longo da obra. O CSV não traz a data de "
                    "assinatura de cada aditivo, então o prazo aditivado daquele dia não é conhecido e a verificação "
                    "usa o prazo original."
                )
        else:
            decorridos_c = max((pd.Timestamp(data_corte) - pd.Timestamp(data_os)).days, 0)
            b1, b2 = st.columns(2)
            b1.metric("Hoje (data de corte)", fmt_data(data_corte))
            b2.metric("Dias decorridos", fmt_dias(decorridos_c))
            b2.caption(f"{100 * decorridos_c / prazo_c:.0f}% do prazo original")
            if grupo == "SUSPENSO_PARALISADO":
                st.warning("Contrato suspenso/paralisado: fica fora do ajuste do modelo. A previsão abaixo é só ilustrativa.")
            st.divider()
            painel_previsao(modelo, data_os, prazo_c, vigente_c if adit_c else None, decorridos_c)
            st.info("Este contrato ainda não terminou. Quando for concluído, ele poderá ser usado aqui para conferir a previsão.")

# --------------------------------------------------------------------------- #
# Aba 3 — Contratos em andamento
# --------------------------------------------------------------------------- #
with aba_and:
    st.subheader("Risco dos contratos em andamento")
    if em_andamento.empty:
        st.info("Nenhum contrato em andamento na base.")
    else:
        st.caption(
            "Probabilidades **condicionais ao tempo já decorrido**: dado que a obra ainda não terminou, "
            "qual a chance de ela terminar **até** o prazo original, até 30 ou até 60 dias depois dele? "
            "Datas que já passaram sem conclusão aparecem como 0%."
        )
        cols = {
            "id_contrato": "Contrato",
            "Numero_contrato": "Nº contrato",
            "Nome_predio": "Prédio",
            "classificacao_contrato": "Tipo",
            "data_os": "OS",
            "prazo_original_dias": "Prazo original (d)",
            "dias_aditivados_app": "Aditivos (d)",
            "prazo_vigente_atual_dias": "Prazo vigente (d)",
            "tempo_observado_dias": "Decorridos (d)",
            "tempo_relativo_observado": "Decorrido / original",
        }
        rotulo_marco = {m: ("P(até o original)" if m == 0 else f"P(até +{m} d)") for m in MARCOS}
        for m in MARCOS:
            cols[f"prob_ate_original_mais_{m}"] = rotulo_marco[m]
        cols.update({
            "prob_estourar_prazo_vigente": "P(passar do vigente)",
            "termino_mediana": "Término P50 (mediana)",
            "termino_p80": "Término P80 (pessimista)",
        })
        tabela = em_andamento[[k for k in cols if k in em_andamento.columns]].rename(columns=cols)
        tabela["OS"] = pd.to_datetime(tabela["OS"]).dt.date
        for k in ("Término P50 (mediana)", "Término P80 (pessimista)"):
            tabela[k] = pd.to_datetime(tabela[k]).dt.date

        barra = st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent")
        config = {"Decorrido / original": st.column_config.NumberColumn(format="%.2f"),
                  "P(passar do vigente)": barra}
        config.update({rotulo_marco[m]: barra for m in MARCOS})
        st.dataframe(tabela, hide_index=True, width="stretch", column_config=config)
        st.download_button(
            "⬇️ Baixar CSV",
            tabela.to_csv(index=False).encode("utf-8-sig"),
            file_name="risco_contratos_em_andamento.csv",
            mime="text/csv",
        )

# --------------------------------------------------------------------------- #
# Aba 4 — Sobre
# --------------------------------------------------------------------------- #
with aba_sobre:
    st.subheader("Como funciona")
    colunas_csv = "`, `".join(core.COLUNAS_OBRIGATORIAS)
    st.markdown(
        f"""
**Target.** `tempo_relativo = duração observada ÷ prazo original`. Assim obras curtas e longas ficam na mesma escala
(1,00 = terminou no prazo; 1,30 = consumiu 30% a mais).

**Modelo.** AFT Log-Normal com uma variável: `log(prazo original)`. Contratos concluídos entram como evento;
contratos em andamento entram como **censurados** (sabemos só que a obra dura *pelo menos* o tempo já decorrido).
Contratos suspensos/paralisados ficam fora do ajuste.

**Ajuste atual:** {modelo.n_eventos_} concluídos, {modelo.n_censurados_} censurados, σ = {modelo.sigma:.2f}, alpha = {alpha}.

### Como ler as probabilidades
- **"Até o prazo original" / "Até 30 dias depois" / "Até 60 dias depois"** são probabilidades **acumuladas**: a chance de
  a obra concluir *no dia ou antes* daquela data. Por isso crescem da esquerda para a direita.
- A chance de terminar *exatamente* num dia é sempre minúscula; por isso o app trabalha com "até".
- As **faixas** mostram a mesma informação por fatias que não se sobrepõem e somam 100%
  (até o prazo · 1 a 30 dias depois · 31 a 60 dias depois · mais de 60 dias depois).
- **P50** é a mediana (50% de chance de terminar até lá); **P80** é o cenário pessimista (80%).
- Tudo é **condicional**: dado que a obra ainda não terminou no dia de hoje. Marcos que já passaram valem 0%.

### Aba "Contratos reais"
Para obras concluídas, o app refaz a previsão num ponto da obra e compara com o término real.
O **modelo da época** usa só o que se sabia naquela data (contratos concluídos depois dela não entram);
o **modelo atual** já viu o desfecho da obra e é otimista. Com poucos concluídos anteriores (menos de
{core.MIN_EVENTOS_PASSADO}), o modelo da época não pode ser ajustado.

**Colunas esperadas no CSV:** `{colunas_csv}`.
(Opcionais: `data_termino_original`, `classificacao_contrato`, `Numero_contrato`, `Nome_predio`, `n_aditivos_prazo` e
os flags da base nova. **`dias_aditivados`**, quando existe, é a soma dos dias da tabela de aditivos de prazo e é usada
no lugar da diferença entre datas.)

### Limitações do piloto
- Base pequena (dezenas de contratos) e **uma única variável**: a previsão depende só do prazo original.
- Probabilidades assumem que o futuro se parece com o passado; não capturam fatores específicos de cada obra
  (chuvas, fornecedor, aditivos em negociação etc.).
- Use como **apoio à priorização**, não como compromisso de data.
"""
    )