"""Piloto: previsão de término de obras (AFT Log-Normal) em Streamlit."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import aft_core as core

DEMO_CSV = Path(__file__).parent / "data" / "contratos_demo.csv"
DATA_CORTE_DEMO = date(2026, 10, 14)

st.set_page_config(page_title="Previsão de término de obras", layout="wide")


# --------------------------------------------------------------------------- #
# Dados e modelo (em cache: o ajuste leva milissegundos, mas evita refazer a cada clique)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def ler_demo() -> pd.DataFrame:
    return pd.read_csv(DEMO_CSV, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def ler_upload(conteudo: bytes) -> pd.DataFrame:
    import io

    return pd.read_csv(io.BytesIO(conteudo), encoding="utf-8-sig", sep=None, engine="python")


@st.cache_data(show_spinner=False)
def ajustar(contratos: pd.DataFrame, data_corte: date, alpha: float):
    df_prep = core.preparar_contratos(contratos, data_corte)
    modelo = core.treinar_modelo(df_prep, alpha)
    return df_prep, modelo


def fmt_pct(x: float) -> str:
    return f"{100 * x:.0f}%"


# --------------------------------------------------------------------------- #
# Barra lateral
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Dados")
    fonte = st.radio(
        "Origem dos contratos",
        [#"Demonstração (dados sintéticos)", 
          "Enviar meu CSV"],
        help="Os dados enviados ficam só na memória da sessão; nada é gravado no repositório.",
    )

    if fonte.startswith("Enviar"):
        arquivo = st.file_uploader("CSV de contratos", type="csv")
        if arquivo is None:
            st.info("Envie o CSV de contratos para começar.")
            st.stop()
        contratos = ler_upload(arquivo.getvalue())
        data_corte_padrao = date.today()
    # else:
    #     contratos = ler_demo()
    #     data_corte_padrao = DATA_CORTE_DEMO

    data_corte = st.date_input(
        "Data de corte (\"hoje\")",
        value=data_corte_padrao,
        help="Contratos sem conclusão até esta data são tratados como censurados (ainda em andamento).",
    )

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

aba_sim, aba_and, aba_sobre = st.tabs(
    ["🔮 Simulador", "📋 Contratos em andamento", "ℹ️ Sobre"]
)

# --------------------------------------------------------------------------- #
# Aba 1 — Simulador
# --------------------------------------------------------------------------- #
with aba_sim:
    st.subheader("Simule uma obra")
    esq, dir_ = st.columns([1, 2], gap="medium")

    with esq:
        prazo = st.number_input("Prazo original (dias)", 30, 3650, 360, step=30)
        decorridos = st.number_input(
            "Dias já decorridos desde a OS", 0, 5000, 0, step=30,
            help="0 = obra ainda não começou. Se já passou do prazo e não terminou, informe aqui.",
        )
        teve_aditivo = st.checkbox("Houve aditivo de prazo?")
        vigente = (
            st.number_input("Prazo vigente (dias, com aditivos)", int(prazo), 5000, int(prazo), step=30)
            if teve_aditivo
            else prazo
        )

    ja_rel = decorridos / prazo
    lim_rel = vigente / prazo
    p_orig = float(modelo.prob_exceder(prazo, ja_rel, 1.0)[0])
    p_vig = float(modelo.prob_exceder(prazo, ja_rel, lim_rel)[0])
    q = {k: float(modelo.quantil_condicional(prazo, ja_rel, k)[0]) * prazo for k in (0.1, 0.5, 0.9)}

    with dir_:
        m1, m2, m3 = st.columns(3)
        m1.metric("Duração P50 (Prob. 50%)", f"{q[0.5]:.0f} dias", f"{q[0.5] / prazo:.2f}× o prazo original", delta_color="off")
        m2.metric("Prob. de passar do prazo original", fmt_pct(p_orig))
        m3.metric(
            "Prob. de passar do prazo vigente" if teve_aditivo else "Prob. de passar do prazo",
            fmt_pct(p_vig),
        )
        st.caption(
            f"Faixa de 80%: a obra deve terminar entre **{q[0.1]:.0f}** e **{q[0.9]:.0f} dias** "
            f"após a OS. Probabilidades condicionais ao fato de a obra *ainda não ter terminado* em {decorridos} dias."
        )

        x_max = max(q[0.9] * 1.15, vigente * 1.3, decorridos * 1.1, prazo * 1.5)
        dias = np.linspace(max(decorridos, 1), x_max, 300)
        cdf = modelo.cdf_condicional(prazo, ja_rel, dias / prazo)
        curva = pd.DataFrame({"dias_desde_a_OS": dias, "prob_acumulada": cdf})

        linhas = [{"dias": prazo, "marca": "Prazo original"}]
        if teve_aditivo and vigente != prazo:
            linhas.append({"dias": vigente, "marca": "Prazo vigente"})
        if decorridos > 0:
            linhas.append({"dias": decorridos, "marca": "Hoje"})
        marcas = pd.DataFrame(linhas)

        base = alt.Chart(curva).mark_line(strokeWidth=3).encode(
            x=alt.X("dias_desde_a_OS:Q", title="Dias desde a OS"),
            y=alt.Y("prob_acumulada:Q", title="Prob. de já ter terminado", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        )
        regras = alt.Chart(marcas).mark_rule(strokeDash=[5, 4]).encode(
            x="dias:Q", color=alt.Color("marca:N", title=None)
        )
        st.altair_chart((base + regras).properties(height=330))

# --------------------------------------------------------------------------- #
# Aba 2 — Contratos em andamento
# --------------------------------------------------------------------------- #
with aba_and:
    st.subheader("Risco dos contratos em andamento")
    if em_andamento.empty:
        st.info("Nenhum contrato em andamento na base.")
    else:
        st.caption(
            "Probabilidades **condicionais ao tempo já decorrido**: dado que a obra ainda não terminou, "
            "qual a chance de ela passar do prazo? Se o prazo já foi ultrapassado sem conclusão, a probabilidade é 100%."
        )
        cols = {
            "id_contrato": "Contrato",
            "classificacao_contrato": "Tipo",
            "data_os": "OS",
            "prazo_original_dias": "Prazo original (d)",
            "prazo_vigente_atual_dias": "Prazo vigente (d)",
            "tempo_observado_dias": "Decorridos (d)",
            "tempo_relativo_observado": "Decorrido / original",
            "prob_estourar_prazo_original": "Prob. (passar do original)",
            "prob_estourar_prazo_vigente": "Prob. (passar do vigente)",
            "termino_mediana": "Término P50 (Prob. 50%)",
            "termino_p80": "Término P80 (Prob. 80%)",
        }
        tabela = em_andamento[[c for c in cols if c in em_andamento.columns]].rename(columns=cols)
        tabela["OS"] = pd.to_datetime(tabela["OS"]).dt.date
        for c in ("Término P50 (Prob. 50%)", "Término P80 (Prob. 80%)"):
            tabela[c] = pd.to_datetime(tabela[c]).dt.date

        st.dataframe(
            tabela,
            hide_index=True,
            width="stretch",
            column_config={
                "Decorrido / original": st.column_config.NumberColumn(format="%.2f"),
                "P(passar do original)": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"),
                "P(passar do vigente)": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"),
            },
        )
        st.download_button(
            "⬇️ Baixar CSV",
            tabela.to_csv(index=False).encode("utf-8-sig"),
            file_name="risco_contratos_em_andamento.csv",
            mime="text/csv",
        )

# --------------------------------------------------------------------------- #
# Aba 3 — Sobre
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

**Colunas esperadas no CSV:** `{colunas_csv}`.
(`data_termino_original` e `classificacao_contrato` são opcionais.)

### Limitações do piloto
- Base pequena (dezenas de contratos) e **uma única variável**: a previsão depende só do prazo original.
- Probabilidades assumem que o futuro se parece com o passado; não capturam fatores específicos de cada obra
  (chuvas, fornecedor, aditivos em negociação etc.).
- Use como **apoio à priorização**, não como compromisso de data.
"""
    )