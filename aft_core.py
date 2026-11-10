"""Núcleo do modelo AFT Log-Normal (portado do notebook 13C).

Sem dependência de scikit-learn: só numpy, pandas e scipy.
A lógica de classificação dos contratos, o target relativo e a validação
temporal seguem exatamente o notebook ``13C_target_relativo_dados_reais``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

# --------------------------------------------------------------------------- #
# Configuração (mesmos valores do notebook)
# --------------------------------------------------------------------------- #
ALPHA_PADRAO = 1.0
ALPHAS = [0.1, 1.0, 10.0, 100.0]

# Marcos de probabilidade: dias DEPOIS do prazo de referência (0 = no próprio prazo).
MARCOS_DIAS = (0, 30, 60)

# Mínimo de obras concluídas para ajustar o "modelo da época" (verificação em contratos reais).
MIN_EVENTOS_PASSADO = 8

FOLD_STARTS = [28, 35, 42, 49]
TAMANHO_FOLD = 7
MIN_FOLD = 5

COLUNAS_DATA = [
    "data_os",
    "data_termino_original",
    "data_termino_vigente_final",
    "data_conclusao_definitiva",
]
COLUNAS_OBRIGATORIAS = [
    "id_contrato",
    "data_os",
    "prazo_original_dias",
    "data_termino_vigente_final",
    "data_conclusao_definitiva",
    "atraso_definitivo_dias",
    "status_final_controle",
]


def colunas_faltantes(df: pd.DataFrame) -> list[str]:
    return [c for c in COLUNAS_OBRIGATORIAS if c not in df.columns]


# --------------------------------------------------------------------------- #
# Preparação dos dados (Etapa 2 do notebook)
# --------------------------------------------------------------------------- #
def preparar_contratos(contratos: pd.DataFrame, data_corte) -> pd.DataFrame:
    """Classifica os contratos e constrói o target relativo ao prazo original."""
    data_corte = pd.Timestamp(data_corte)
    df = contratos.copy()

    for col in COLUNAS_DATA:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    df["prazo_original_dias"] = pd.to_numeric(df["prazo_original_dias"], errors="coerce")
    df["atraso_definitivo_dias"] = pd.to_numeric(df["atraso_definitivo_dias"], errors="coerce")

    concluido = df["data_conclusao_definitiva"].notna() & df["atraso_definitivo_dias"].notna()
    suspenso = (
        df["status_final_controle"].astype(str).str.upper().str.contains("SUSPENSO|PARALISADO")
    )

    df["grupo_modelagem"] = np.select(
        [concluido, suspenso],
        ["CONCLUIDO_COM_TARGET", "SUSPENSO_PARALISADO"],
        default="EM_ANDAMENTO_CENSURADO",
    )
    df["evento_conclusao"] = concluido.astype(int)

    # Linhas sem data de OS ou sem prazo válido não conseguem entrar no modelo.
    valido = df["data_os"].notna() & (df["prazo_original_dias"] > 0)
    df["flag_apto_modelagem"] = (
        (~df["grupo_modelagem"].eq("SUSPENSO_PARALISADO")) & valido
    ).astype(int)

    df["data_fim_observacao"] = df["data_conclusao_definitiva"].where(concluido, data_corte)
    df["tempo_observado_dias"] = (df["data_fim_observacao"] - df["data_os"]).dt.days.clip(lower=1)
    df["tempo_relativo_observado"] = df["tempo_observado_dias"] / df["prazo_original_dias"]
    df["log_prazo_original"] = np.log1p(df["prazo_original_dias"])

    if "data_termino_vigente_final" in df.columns:
        prazo_vigente = (df["data_termino_vigente_final"] - df["data_os"]).dt.days
        df["prazo_vigente_atual_dias"] = prazo_vigente.where(
            prazo_vigente > 0, df["prazo_original_dias"]
        )
    else:
        df["prazo_vigente_atual_dias"] = df["prazo_original_dias"]
    df["limite_relativo_vigente"] = df["prazo_vigente_atual_dias"] / df["prazo_original_dias"]
    return df


# --------------------------------------------------------------------------- #
# Modelo
# --------------------------------------------------------------------------- #
class Padronizador:
    """Imputação pela mediana + padronização (equivale ao pipeline do notebook)."""

    def fit(self, x):
        x = np.asarray(x, dtype=float)
        self.mediana_ = np.nanmedian(x)
        x = np.where(np.isnan(x), self.mediana_, x)
        self.media_ = x.mean()
        dp = x.std()  # ddof=0, igual ao StandardScaler
        self.escala_ = dp if dp > 0 else 1.0
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=float)
        x = np.where(np.isnan(x), self.mediana_, x)
        return (x - self.media_) / self.escala_


class LogNormalAFT:
    """AFT Log-Normal com penalização L2 nos coeficientes (intercepto livre)."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    @staticmethod
    def _com_intercepto(X):
        X = np.asarray(X, dtype=float)
        return np.column_stack([np.ones(len(X)), X])

    def fit(self, X, tempo, evento):
        X1 = self._com_intercepto(X)
        y = np.log(np.maximum(np.asarray(tempo, dtype=float), 1e-8))
        evento = np.asarray(evento, dtype=int)
        p = X1.shape[1]

        def neg_loglik(theta):
            beta, sigma = theta[:p], np.exp(theta[p])
            z = (y - X1 @ beta) / sigma
            loglik = np.where(
                evento == 1,
                norm.logpdf(z) - np.log(sigma) - y,  # concluído: densidade
                norm.logsf(z),                       # censurado: P(T > t observado)
            )
            return -loglik.sum() + 0.5 * self.alpha * np.sum(beta[1:] ** 2)

        inicial = np.r_[np.median(y), np.zeros(p - 1), np.log(np.std(y) + 1e-3)]
        self.resultado_ = minimize(
            neg_loglik, inicial, method="L-BFGS-B", bounds=[(None, None)] * p + [(-5, 3)]
        )
        self.beta_, self.sigma_ = self.resultado_.x[:p], np.exp(self.resultado_.x[p])
        return self

    def mu(self, X):
        return self._com_intercepto(X) @ self.beta_

    def sobrevivencia(self, X, tempo_relativo):
        """P(T > t)."""
        t = np.maximum(np.asarray(tempo_relativo, dtype=float), 1e-8)
        return norm.sf((np.log(t) - self.mu(X)) / self.sigma_)


class ModeloObras:
    """Envolve padronização + AFT e expõe tudo em função do prazo original (dias)."""

    def __init__(self, alpha: float = ALPHA_PADRAO):
        self.alpha = alpha

    def fit(self, prazo_original_dias, tempo_relativo, evento):
        log_prazo = np.log1p(np.asarray(prazo_original_dias, dtype=float))
        self.pad_ = Padronizador().fit(log_prazo)
        self.aft_ = LogNormalAFT(self.alpha).fit(
            self.pad_.transform(log_prazo).reshape(-1, 1), tempo_relativo, evento
        )
        self.n_eventos_ = int(np.sum(evento))
        self.n_censurados_ = int(len(evento) - np.sum(evento))
        return self

    # -- utilidades ----------------------------------------------------------
    def _X(self, prazo):
        prazo = np.atleast_1d(np.asarray(prazo, dtype=float))
        return self.pad_.transform(np.log1p(prazo)).reshape(-1, 1)

    @property
    def sigma(self) -> float:
        return float(self.aft_.sigma_)

    @property
    def convergiu(self) -> bool:
        return bool(self.aft_.resultado_.success)

    # -- previsões -----------------------------------------------------------
    def mediana_relativa(self, prazo):
        return np.exp(self.aft_.mu(self._X(prazo)))

    def sobrevivencia(self, prazo, t_rel):
        return self.aft_.sobrevivencia(self._X(prazo), t_rel)

    def prob_exceder(self, prazo, ja_observado_rel, limite_rel):
        """P(T > limite | T > já observado). Se o limite já passou sem conclusão → 1."""
        prazo = np.atleast_1d(np.asarray(prazo, dtype=float))
        ja = np.broadcast_to(np.asarray(ja_observado_rel, dtype=float), prazo.shape)
        lim = np.broadcast_to(np.asarray(limite_rel, dtype=float), prazo.shape)
        prob = self.sobrevivencia(prazo, lim) / np.maximum(self.sobrevivencia(prazo, ja), 1e-12)
        return np.clip(np.where(lim <= ja, 1.0, prob), 0.0, 1.0)

    def quantil_condicional(self, prazo, ja_observado_rel, q):
        """Tempo relativo x tal que P(T <= x | T > já observado) = q."""
        prazo = np.atleast_1d(np.asarray(prazo, dtype=float))
        ja = np.broadcast_to(np.asarray(ja_observado_rel, dtype=float), prazo.shape)
        s0 = np.maximum(self.sobrevivencia(prazo, ja), 1e-12)
        z = norm.isf(np.clip((1.0 - q) * s0, 1e-300, 1 - 1e-12))
        return np.exp(self.aft_.mu(self._X(prazo)) + self.aft_.sigma_ * z)

    def cdf_condicional(self, prazo, ja_observado_rel, t_rel):
        """P(T <= t | T > já observado), para um único contrato e vários t."""
        t_rel = np.asarray(t_rel, dtype=float)
        s_t = self.aft_.sobrevivencia(self._X(prazo), t_rel)  # (n_t,)
        s_0 = max(float(self.sobrevivencia(prazo, ja_observado_rel)[0]), 1e-12)
        cdf = 1.0 - s_t / s_0
        return np.where(t_rel <= ja_observado_rel, 0.0, np.clip(cdf, 0.0, 1.0))


def treinar_modelo(df_prep: pd.DataFrame, alpha: float = ALPHA_PADRAO) -> ModeloObras:
    """Ajuste final: todos os contratos aptos (concluídos = evento, abertos = censurados)."""
    base = df_prep[df_prep["flag_apto_modelagem"].eq(1)]
    return ModeloObras(alpha).fit(
        base["prazo_original_dias"], base["tempo_relativo_observado"], base["evento_conclusao"]
    )


# --------------------------------------------------------------------------- #
# Contratos em andamento (Etapa 7 do notebook, com probabilidades condicionais)
# --------------------------------------------------------------------------- #
def prever_em_andamento(df_prep: pd.DataFrame, modelo: ModeloObras) -> pd.DataFrame:
    a = df_prep[
        df_prep["grupo_modelagem"].eq("EM_ANDAMENTO_CENSURADO") & df_prep["flag_apto_modelagem"].eq(1)
    ].copy()
    if a.empty:
        return a

    prazo = a["prazo_original_dias"].to_numpy(float)
    ja = a["tempo_relativo_observado"].to_numpy(float)

    a["prob_estourar_prazo_original"] = modelo.prob_exceder(prazo, ja, 1.0)
    a["prob_estourar_prazo_vigente"] = modelo.prob_exceder(
        prazo, ja, a["limite_relativo_vigente"].to_numpy(float)
    )
    # P(terminar até o prazo original + m dias), para cada marco m.
    decorridos = a["tempo_observado_dias"].to_numpy(float)
    for m in MARCOS_DIAS:
        a[f"prob_ate_original_mais_{m}"] = prob_ate(modelo, prazo, decorridos, prazo + m)
    mediana = modelo.quantil_condicional(prazo, ja, 0.5) * prazo
    p80 = modelo.quantil_condicional(prazo, ja, 0.8) * prazo
    a["duracao_mediana_dias"] = mediana
    a["termino_mediana"] = a["data_os"] + pd.to_timedelta(np.round(mediana), unit="D")
    a["termino_p80"] = a["data_os"] + pd.to_timedelta(np.round(p80), unit="D")
    return a.sort_values("prob_estourar_prazo_vigente", ascending=False)


# --------------------------------------------------------------------------- #
# Validação temporal (Etapas 3 e 4 do notebook; baseline + AFT)
# --------------------------------------------------------------------------- #
def c_index(tempo, evento, risco):
    t, e, r = (np.asarray(x, dtype=float) for x in (tempo, evento, risco))
    comparavel = (t[:, None] < t[None, :]) & (e[:, None] == 1)
    if not comparavel.any():
        return np.nan
    concordante = comparavel & (r[:, None] > r[None, :])
    empate = comparavel & (r[:, None] == r[None, :])
    return (concordante.sum() + 0.5 * empate.sum()) / comparavel.sum()


def _treino_no_passado(base: pd.DataFrame, cutoff) -> pd.DataFrame:
    train = base[base["data_os"] <= cutoff].copy()
    evento = train["evento_conclusao"].eq(1) & (train["data_conclusao_definitiva"] <= cutoff)
    fim = train["data_conclusao_definitiva"].where(evento, cutoff)
    train["evento_fold"] = evento.astype(int)
    train["tempo_rel_fold"] = (fim - train["data_os"]).dt.days.clip(lower=1) / train[
        "prazo_original_dias"
    ]
    return train


def _metricas(val: pd.DataFrame, pred_rel) -> dict:
    pred_rel = np.asarray(pred_rel, dtype=float)
    ok = val["evento_conclusao"].eq(1).to_numpy()
    y_rel = val["tempo_relativo_observado"].to_numpy()[ok]
    prazo = val["prazo_original_dias"].to_numpy()[ok]
    y_dias = val["tempo_observado_dias"].to_numpy()[ok]
    p_rel = pred_rel[ok]
    p_dias = p_rel * prazo
    return {
        "mae_duracao_dias": float(np.mean(np.abs(y_dias - p_dias))),
        "mae_relativo": float(np.mean(np.abs(y_rel - p_rel))),
        "cindex": c_index(y_rel, np.ones(ok.sum()), -p_rel),
        "cindex_censurado": c_index(
            val["tempo_relativo_observado"], val["evento_conclusao"], -pred_rel
        ),
    }


def validacao_temporal(
    df_prep: pd.DataFrame,
    alpha: float = ALPHA_PADRAO,
    fold_starts=FOLD_STARTS,
    tamanho_fold: int = TAMANHO_FOLD,
    min_fold: int = MIN_FOLD,
) -> pd.DataFrame:
    """Retorna uma linha por (fold, modelo) com as métricas de validação."""
    model_data = (
        df_prep[df_prep["flag_apto_modelagem"].eq(1)]
        .sort_values(["data_os", "id_contrato"])
        .reset_index(drop=True)
    )
    linhas = []
    fold_id = 0
    for start in fold_starts:
        val = model_data.iloc[start : start + tamanho_fold].copy()
        if len(val) < min_fold:
            continue
        cutoff = val["data_os"].min() - pd.Timedelta(days=1)
        train = _treino_no_passado(model_data.iloc[:start], cutoff)
        concl = train[train["evento_fold"].eq(1)]
        if len(concl) < 5:
            continue
        fold_id += 1
        info = {
            "fold": fold_id,
            "cutoff": cutoff.date(),
            "eventos_treino": int(train["evento_fold"].sum()),
            "censurados_treino": int(train["evento_fold"].eq(0).sum()),
            "concluidos_validacao": int(val["evento_conclusao"].sum()),
        }

        pred_base = np.full(len(val), concl["tempo_rel_fold"].median())
        linhas.append({**info, "modelo": "Baseline (mediana)", **_metricas(val, pred_base)})

        m = ModeloObras(alpha).fit(
            train["prazo_original_dias"], train["tempo_rel_fold"], train["evento_fold"]
        )
        pred_aft = m.mediana_relativa(val["prazo_original_dias"].to_numpy(float))
        linhas.append({**info, "modelo": "AFT Log-Normal", **_metricas(val, pred_aft)})
    return pd.DataFrame(linhas)


# --------------------------------------------------------------------------- #
# Marcos de probabilidade, faixas e previsão por contrato (usados pelo app)
# --------------------------------------------------------------------------- #
def prob_ate(modelo: ModeloObras, prazo_original, dias_decorridos, limite_dias):
    """P(a obra terminar até `limite_dias` após a OS | ainda não terminou em `dias_decorridos`).

    "Até" = no dia ou antes (probabilidade acumulada). Se o limite já ficou para trás sem a obra
    ter terminado, a probabilidade é 0. Aceita escalares ou arrays (um item por contrato).
    """
    prazo = np.atleast_1d(np.asarray(prazo_original, dtype=float))
    ja = np.asarray(dias_decorridos, dtype=float) / prazo
    lim = np.asarray(limite_dias, dtype=float) / prazo
    return 1.0 - modelo.prob_exceder(prazo, ja, lim)


def nomes_faixas(marcos=MARCOS_DIAS) -> list:
    """Nomes das faixas entre marcos consecutivos. Com (0, 30, 60):
    'Até o prazo', '1 a 30 dias depois', '31 a 60 dias depois', 'Mais de 60 dias depois'."""
    nomes = ["Até o prazo" if marcos[0] == 0 else f"Até {marcos[0]} dias depois"]
    for ant, atual in zip(marcos[:-1], marcos[1:]):
        nomes.append(f"{ant + 1} a {atual} dias depois")
    nomes.append(f"Mais de {marcos[-1]} dias depois")
    return nomes


def faixa_do_atraso(atraso_dias: float, marcos=MARCOS_DIAS) -> int:
    """Índice (em `nomes_faixas`) da faixa em que cai um término `atraso_dias` após o prazo de referência."""
    for i, m in enumerate(marcos):
        if atraso_dias <= m:
            return i
    return len(marcos)


def faixas_por_marco(modelo: ModeloObras, prazo_original, dias_decorridos, limite_dias, marcos=MARCOS_DIAS) -> list:
    """Probabilidade de a obra terminar em cada faixa (somam 1): lista de (nome, prob)."""
    acum = [float(prob_ate(modelo, prazo_original, dias_decorridos, limite_dias + m)[0]) for m in marcos]
    partes = [acum[0]] + [max(b - a, 0.0) for a, b in zip(acum[:-1], acum[1:])] + [max(1.0 - acum[-1], 0.0)]
    return list(zip(nomes_faixas(marcos), partes))


def previsao_contrato(modelo: ModeloObras, prazo_original, dias_decorridos, prazo_vigente=None, marcos=MARCOS_DIAS) -> dict:
    """Tudo o que o app mostra para um contrato, condicional a a obra ainda não ter terminado.

    prazo_vigente: prazo em dias contados da OS (original + aditivos). None = sem aditivo conhecido.
    """
    prazo = float(prazo_original)
    decorridos = float(dias_decorridos)
    vigente = float(prazo_vigente) if prazo_vigente else prazo
    ja = decorridos / prazo

    def _marcos(limite):
        return [
            {"offset": m, "limite_dias": limite + m,
             "prob_ate": float(prob_ate(modelo, prazo, decorridos, limite + m)[0])}
            for m in marcos
        ]

    quantis = {
        f"p{int(round(q * 100))}": float(modelo.quantil_condicional(prazo, ja, q)[0]) * prazo
        for q in (0.1, 0.5, 0.8, 0.9)
    }
    return {
        "dias_decorridos": decorridos,
        "prazo_original": prazo,
        "prazo_vigente": vigente,
        "quantis": quantis,
        "original": _marcos(prazo),
        "vigente": _marcos(vigente) if vigente > prazo else None,
        "faixas": faixas_por_marco(modelo, prazo, decorridos, prazo, marcos),
        "prob_estourar_original": float(modelo.prob_exceder(prazo, ja, 1.0)[0]),
        "prob_estourar_vigente": float(modelo.prob_exceder(prazo, ja, vigente / prazo)[0]),
    }


def treinar_no_passado(df_prep: pd.DataFrame, cutoff, alpha: float = ALPHA_PADRAO,
                       min_eventos: int = MIN_EVENTOS_PASSADO):
    """Modelo como ele seria ajustado na data `cutoff`, usando só o que se sabia até então.

    Entram os contratos com OS até `cutoff`; quem não tinha concluído até `cutoff` entra como
    censurado (mesmo que hoje já tenha terminado). Retorna (modelo, info); modelo = None quando
    havia menos de `min_eventos` obras concluídas ou o ajuste não convergiu.
    """
    cutoff = pd.Timestamp(cutoff)
    base = df_prep[df_prep["flag_apto_modelagem"].eq(1)]
    train = _treino_no_passado(base, cutoff)
    info = {
        "cutoff": cutoff.date(),
        "eventos": int(train["evento_fold"].sum()),
        "censurados": int(train["evento_fold"].eq(0).sum()),
    }
    if info["eventos"] < min_eventos:
        return None, info
    modelo = ModeloObras(alpha).fit(
        train["prazo_original_dias"], train["tempo_rel_fold"], train["evento_fold"]
    )
    return (modelo if modelo.convergiu else None), info
