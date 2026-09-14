# Projeto Semantix — Previsão de Admissão/Desligamento (CAGED/RAIS)

Projeto de conclusão de módulo do curso de Cientista de Dados da Ebac, desenvolvido para a empresa Semantix.

## Objetivo

Prever, a partir de registros do mercado de trabalho brasileiro (CAGED/RAIS), se uma movimentação trabalhista corresponde a uma **admissão** ou a um **desligamento** (coluna alvo `admitidos_desligados`).

## Dados

Dados de 2007 a 2019 (amostra de 10 mil registros por ano), carregados via `basedosdados`/BigQuery, com colunas como `tipo_movimentacao_desagregado`, `tipo_estabelecimento`, `faixa_emprego_inicio_janeiro`, `cbo_2002`, `salario_mensal`, `idade`, `sexo`, `raca_cor`, entre outras. Uma fatia de 2018/2019 (`dados_futuros.csv`) foi reservada desde o início como conjunto de teste final, nunca usado em treinamento.

## Etapas realizadas

1. **Leitura e exploração inicial** dos dados via BigQuery.
2. **Tratamento de dados**: conversão de tipos, tratamento da coluna `tipo_movimentacao_desagregado` (incluindo o valor 90 remapeado para 33).
3. **Análise exploratória**: distribuição por sexo, raça/cor, salário ao longo dos anos, grau de instrução, e análise de correlação — identificando `tipo_movimentacao_desagregado` como a variável mais preditiva (correlação de -0,37 com o alvo).
4. **Modelagem comparativa**: 4 algoritmos (Rede Neural, Random Forest, Naive Bayes, XGBoost), cada um em duas versões — **base** (tratamento mínimo) e **aprimorada** (Pipeline completo com `OneHotEncoder` e `StandardScaler`).
5. **Escolha do modelo final** e salvamento como artefato reutilizável (`modelo_xgb_final.pkl`, via `joblib`).
6. **Validação final em dados nunca vistos**: o modelo foi aplicado sobre `dados_futuros.csv` (2018/2019, fora do período de treino), gerando `dados_futuros_resultado.csv`.
7. **Script de geração do aplicativo de previsão** (`prever.py`): código-fonte com interface gráfica (Tkinter, com suporte a arrastar-e-soltar) que carrega o modelo salvo, roda a previsão sobre um CSV informado pelo usuário e exibe acurácia, relatório de classificação e matriz de confusão. A partir dele foi gerado o executável `prever.exe`, disponível na aba Releases do repositório.
8. **Dashboard interativo (Power BI)**: painel com 2 páginas (Visão geral e Previsões 2018/2019), consolidando visualmente os principais achados da análise e da modelagem.

## Resultados

| Modelo | Experimento | Acurácia | Erros (de 21.769) |
|---|---|---|---|
| Rede Neural | Base | 96,10% | 848 |
| Rede Neural | Aprimorado | 98,98% | 223 |
| Random Forest | Base | 99,84% | 34 |
| Random Forest | Aprimorado | 100,00% | 0 |
| Naive Bayes | Base | 58,96% | 8935 |
| Naive Bayes | Aprimorado | 100,00% | 0 |
| XGBoost | Base | 100,00% | 0 |
| XGBoost | Aprimorado | 100,00% | 0 |

O **XGBoost** (versão aprimorada) foi escolhido como modelo final. Aplicado ao conjunto `dados_futuros.csv` (2018/2019, nunca visto durante o treinamento), obteve **100% de acurácia** nos 19.649 registros avaliados, confirmando que o modelo generaliza bem e não apenas memorizou os dados de treino.

O desempenho quase perfeito de todos os algoritmos se deve à variável `tipo_movimentacao_desagregado`, cuja forte correlação com o alvo torna a separação entre admissão e desligamento praticamente trivial para qualquer modelo que a utilize sem descartá-la no pré-processamento.

## Tecnologias

- Python, pandas, BigQuery (`basedosdados`)
- scikit-learn (ColumnTransformer, OneHotEncoder, StandardScaler)
- XGBoost, Random Forest, Naive Bayes, Rede Neural (MLP)
- joblib (persistência do modelo)
- Tkinter + tkinterdnd2 (interface gráfica do aplicativo de previsão, compilado em executável)
- Power BI (dashboard interativo)

## Documentos do repositório

Guia rápido do que cada arquivo contém, para quem está vendo o projeto pela primeira vez:

- **`Projeto_Semantix.ipynb`** — notebook principal do projeto. Contém toda a análise: carregamento e limpeza dos dados, análise exploratória, extração de insights, treinamento e comparação dos 4 modelos, e o salvamento do modelo final.
- **`Apresentacao_Projeto_Semantix.pptx`** — apresentação em slides com o resumo executivo do projeto (contexto, principais insights e resultados), pensada para expor o trabalho sem precisar abrir o notebook.
- **`Apresentacao_Projeto_Semantix.pdf`** — exportação em PDF da apresentação acima, para quem quer ver os slides sem precisar abrir no PowerPoint.
- **`Dashboard Semantix.pbix`** — arquivo do dashboard interativo em Power BI, com 2 páginas (Visão geral e Previsões 2018/2019). Requer o Power BI Desktop (gratuito) para ser aberto e explorado interativamente.
- **`Dashboard Semantix.pdf`** — exportação em PDF do dashboard acima, com uma página por página do painel. Permite visualizar o resultado do dashboard sem precisar instalar o Power BI.
- **`prever.py`** — script-fonte (Python) do aplicativo de previsão com interface gráfica (arrastar-e-soltar). Carrega o modelo já treinado (`modelo_xgb_final.pkl`) e aplica a previsão sobre um novo CSV, exibindo acurácia, relatório de classificação e matriz de confusão. Não é o aplicativo pronto — é o código a partir do qual o executável abaixo foi gerado.
- **`prever.exe`** (disponível na aba **Releases** do repositório, não faz parte do código-fonte) — versão compilada e pronta para uso do aplicativo de previsão acima, sem precisar instalar Python ou dependências.
- **`modelo_xgb_final.pkl`** — modelo XGBoost final, já treinado e salvo (via `joblib`), pronto para ser reutilizado pelo `prever.py` sem precisar re-treinar.
- **`dados_futuros.csv`** — conjunto de dados de 2018/2019, reservado desde o início como teste final (nunca usado no treinamento), usado para validar a generalização do modelo.
- **`dados_futuros_resultado.csv`** — resultado da aplicação do modelo final sobre `dados_futuros.csv`, com as previsões geradas.

## Como executar

1. Instale as dependências: `pip install pandas scikit-learn xgboost joblib basedosdados tkinterdnd2`.
2. Execute `Projeto_Semantix.ipynb` para reproduzir a análise e o treinamento (gera `modelo_xgb_final.pkl`).
3. Para aplicar o modelo já treinado a um novo CSV sem instalar Python, baixe `prever.exe` na aba **Releases** do repositório. Alternativamente, rode o script-fonte: `python prever.py` (abre uma janela para arrastar o arquivo) ou `python prever.py caminho/para/arquivo.csv`. O CSV deve conter a coluna `admitidos_desligados` para o cálculo das métricas.
4. Para explorar o dashboard interativamente, abra `Dashboard Semantix.pbix` no Power BI Desktop. Para uma visualização rápida sem instalar nada, veja `Dashboard Semantix.pdf`.
