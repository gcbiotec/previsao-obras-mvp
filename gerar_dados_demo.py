"""Gera data/contratos_demo.csv: contratos SINTÉTICOS com o mesmo formato da base real.

Não contém nenhum dado real. Serve para demonstrar o app sem expor contratos.
Uso: python gerar_dados_demo.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
DATA_CORTE = pd.Timestamp("2026-10-14")
N = 64

classes = ["Construção", "Reforma Global", "Reforma Parcial", "Reforma Global e Ampliação"]
prazos_por_classe = {
    "Construção": [360, 540, 600, 720, 1080, 1260],
    "Reforma Global": [240, 360, 540, 900],
    "Reforma Parcial": [90, 120, 150, 180, 240, 540],
    "Reforma Global e Ampliação": [360, 660, 720],
}

inicio, fim = pd.Timestamp("2019-01-01"), pd.Timestamp("2026-08-15")
dias_os = np.sort(rng.integers(0, (fim - inicio).days, size=N))

linhas = []
for i, d in enumerate(dias_os):
    classe = rng.choice(classes, p=[0.25, 0.25, 0.35, 0.15])
    prazo = int(rng.choice(prazos_por_classe[classe]))
    data_os = inicio + pd.Timedelta(days=int(d))

    # Tempo relativo ~ log-normal; obras longas tendem a estourar um pouco menos em proporção.
    mu = 0.42 - 0.10 * (np.log(prazo) - 5.7)
    rel = float(np.clip(np.exp(rng.normal(mu, 0.35)), 0.55, 6.0))
    duracao = max(int(round(rel * prazo)), 30)
    conclusao = data_os + pd.Timedelta(days=duracao)

    # Aditivos de prazo: em ~45% dos casos o prazo vigente cobre parte do estouro.
    termino_original = data_os + pd.Timedelta(days=prazo)
    vigente = termino_original
    if rng.random() < 0.45 and duracao > prazo:
        vigente = termino_original + pd.Timedelta(days=int((duracao - prazo) * rng.uniform(0.3, 1.0)))

    em_andamento = conclusao > DATA_CORTE
    linhas.append(
        {
            "id_contrato": 100 + i * 6 + int(rng.integers(0, 5)),
            "classificacao_contrato": classe,
            "data_os": data_os.date(),
            "prazo_original_dias": prazo,
            "data_termino_original": termino_original.date(),
            "data_termino_vigente_final": vigente.date(),
            "data_conclusao_definitiva": None if em_andamento else conclusao.date(),
            "atraso_definitivo_dias": None if em_andamento else (conclusao - termino_original).days,
            "status_final_controle": "EM EXECUÇÃO" if em_andamento else "CONCLUÍDO",
        }
    )

df = pd.DataFrame(linhas)

# Um contrato suspenso (fica fora do ajuste, como na base real).
idx = df.index[df["data_conclusao_definitiva"].isna()]
if len(idx):
    df.loc[idx[0], "status_final_controle"] = "SUSPENSO"

saida = Path(__file__).parent / "data" / "contratos_demo.csv"
saida.parent.mkdir(exist_ok=True)
df.to_csv(saida, index=False, encoding="utf-8-sig")
print(f"{len(df)} contratos -> {saida}")
print(df["status_final_controle"].value_counts().to_string())
