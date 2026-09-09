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
7. **Ferramenta de uso** (`prever.py`): aplicativo com interface gráfica (Tkinter, com suporte a arrastar-e-soltar) que carrega o modelo salvo, roda a previsão sobre um CSV informado pelo usuário e exibe acurácia, relatório de classificação e matriz de confusão.

## Resultados

| Modelo | Experimento | Acurácia | Erros (de 21.769) |
|---|---|---|---|
| Rede Neural | Base | 96,10% | 848 |
| Rede Neural | Aprimorado | 98,98% | 223 |
| Random Forest | Base | 99,84% | 34 |
| Random Forest | Aprimorado | 100,00% | 0 |
| Naive Bayes | Base | 58,96% | 8935 |
| Naive Bayes | Aprimorado | 100,00% | 1 |
| XGBoost | Base | 100,00% | 0 |
| XGBoost | Aprimorado | 100,00% | 0 |

O **XGBoost** (versão aprimorada) foi escolhido como modelo final. Aplicado ao conjunto `dados_futuros.csv` (2018/2019, nunca visto durante o treinamento), obteve **100% de acurácia** nos 19.649 registros avaliados, confirmando que o modelo generaliza bem e não apenas memorizou os dados de treino.

O desempenho quase perfeito de todos os algoritmos se deve à variável `tipo_movimentacao_desagregado`, cuja forte correlação com o alvo torna a separação entre admissão e desligamento praticamente trivial para qualquer modelo que a utilize sem descartá-la no pré-processamento.

## Tecnologias

- Python, pandas, BigQuery (`basedosdados`)
- scikit-learn (ColumnTransformer, OneHotEncoder, StandardScaler)
- XGBoost, Random Forest, Naive Bayes, Rede Neural (MLP)
- joblib (persistência do modelo)
- Tkinter + tkinterdnd2 (aplicativo de previsão com interface gráfica)

## Como executar

1. Instale as dependências: `pip install pandas scikit-learn xgboost joblib basedosdados tkinterdnd2`.
2. Execute `Projeto_Semantix.ipynb` para reproduzir a análise e o treinamento (gera `modelo_xgb_final.pkl`).
3. Para aplicar o modelo já treinado a um novo CSV, rode `python prever.py` (abre uma janela para arrastar o arquivo) ou `python prever.py caminho/para/arquivo.csv`. O CSV deve conter a coluna `admitidos_desligados` para o cálculo das métricas.
