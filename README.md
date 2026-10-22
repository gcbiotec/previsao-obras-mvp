# Previsão de término de obras — piloto (MVP)

App Streamlit que usa o modelo **AFT Log-Normal** do notebook `13C_target_relativo` para estimar
quanto do prazo original uma obra tende a consumir e qual a chance de ela passar do prazo.

| Aba | Para quê |
|---|---|
| Simulador | Informe prazo, dias decorridos e aditivos → mediana, faixa de 80% e probabilidade de estourar |
| Contratos em andamento | Ranking de risco com data estimada de término (mediana e cenário pessimista) |
| Validação | Compara o AFT com o baseline (mediana) na validação temporal do notebook |
| Sobre | Método e limitações |

## Rodar localmente

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Testes: `pip install -r requirements-dev.txt && pytest -q`

## Estrutura

```
app.py                  interface Streamlit
aft_core.py             modelo, target, previsões e validação (só numpy/pandas/scipy)
gerar_dados_demo.py     gera o CSV sintético
data/contratos_demo.csv dados SINTÉTICOS (únicos dados versionados)
tests/test_app.py       teste de fumaça + checagens numéricas
```

## Dados

- O repositório traz apenas dados **sintéticos**. O `.gitignore` bloqueia qualquer outro CSV.
- Para usar a base real, escolha **"Enviar meu CSV"** na barra lateral. O arquivo fica na memória da sessão
  (não é gravado no repositório), mas **trafega pelos servidores da plataforma de hospedagem**. Confirme com
  a área de dados/segurança se isso é aceitável antes de subir dados reais num serviço público.
- Colunas obrigatórias: `id_contrato, data_os, prazo_original_dias, data_termino_vigente_final,
  data_conclusao_definitiva, atraso_definitivo_dias, status_final_controle`.

## Deploy gratuito (Streamlit Community Cloud)

O GitHub guarda o código; quem **executa** o app é o Streamlit Community Cloud (o GitHub Pages só serve
páginas estáticas e não roda Python).

1. Crie um repositório no GitHub e envie estes arquivos.
2. Acesse <https://share.streamlit.io>, entre com o GitHub e clique em **Create app**.
3. Escolha o repositório, a branch e `app.py` como arquivo principal.
4. **Deploy.** A URL pública fica pronta em poucos minutos; cada `git push` atualiza o app.

Repositório público → app público (qualquer pessoa com o link acessa). Repositório privado → app privado,
visível só para convidados por e-mail. Na versão gratuita, só 1 app privado por vez.
