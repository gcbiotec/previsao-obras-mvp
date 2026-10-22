"""Teste de fumaça do app + checagens numéricas do núcleo.  Rodar: pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import aft_core as core  # noqa: E402


def test_app_abre_sem_erros():
    at = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 4
    assert at.metric[0].value == "64"  # contratos na base


def test_simulador_responde_a_entradas():
    at = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=60).run()
    at.number_input[0].set_value(720).run()  # prazo original
    at.number_input[1].set_value(900).run()  # dias decorridos (já passou do prazo)
    assert not at.exception
    valores = {m.label: m.value for m in at.metric}
    assert valores["Prob. de passar do prazo original"] == "100%"


def test_probabilidades_coerentes():
    df = pd.read_csv(RAIZ / "data" / "contratos_demo.csv")
    prep = core.preparar_contratos(df, "2026-10-14")
    modelo = core.treinar_modelo(prep)
    assert modelo.convergiu
    # CDF condicional é monotônica e começa em zero
    t = np.linspace(0.2, 5, 200)
    cdf = modelo.cdf_condicional(360, 0.2, t)
    assert np.all(np.diff(cdf) >= -1e-12) and cdf[0] == 0
    # quantil e CDF são inversos
    q80 = modelo.quantil_condicional(360, 0.5, 0.8)
    assert abs(modelo.cdf_condicional(360, 0.5, q80)[0] - 0.8) < 1e-6
    # prob. de estourar nunca diminui se o limite diminui
    assert modelo.prob_exceder(360, 0.3, 1.0)[0] >= modelo.prob_exceder(360, 0.3, 1.5)[0]
    # em andamento: quem já passou do limite tem prob. 1
    a = core.prever_em_andamento(prep, modelo)
    ja_passou = a["tempo_relativo_observado"] >= a["limite_relativo_vigente"]
    assert (a.loc[ja_passou, "prob_estourar_prazo_vigente"] == 1.0).all()
