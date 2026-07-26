# 📊 Portfólio de Data Science & Machine Learning

Bem-vindo(a) ao meu repositório de projetos de Aprendizado de Máquina aplicados a soluções financeiras e imobiliárias. Este repositório contém dois projetos  cobrindo as tarefas fundamentais de **Regressão** e **Classificação**, focados na resolução de problemas reais de negócio através da análise de dados e modelagem preditiva.

---

## 📂 Visão Geral dos Projetos

### 🏡 1. Previsão de Preços Imobiliários (Ames Housing) — *Regressão*
* **Objetivo de Negócio:** Construir um Modelo de Avaliação Automatizada (AVM - *Automated Valuation Model*) para precificar imóveis residenciais com alta precisão e baixa latência.
* **Desafio Técnico:** Tratar um dataset de alta dimensionalidade (mais de 200 variáveis originais/dummies), aplicando engenharia de recursos, remoção de *outliers* e seleção de atributos para produção.
* **Principais Resultados:**
  * Redução de **81% na quantidade de variáveis** (de 211 para 40 variáveis) utilizando *Sequential Feature Selector* (SFS).
  * Manutenção do poder explicativo do modelo ($R^2 \approx 86\%$) com impacto mínimo nas métricas de erro.
  * **MAE Final:** ~$17.555 USD (representando uma margem de erro relativa de ~10%, perfeitamente alinhada com a margem natural de barganha do mercado imobiliário).
* **Tecnologias:** `Python`, `Scikit-Learn`, `Pandas`, `PowerTransformer`, `Pipelines/ColumnTransformer`.

---

### 💳 2. Avaliação de Risco de Crédito (German Credit Data) — *Classificação Binária*
* **Objetivo de Negócio:** Prever a probabilidade de inadimplência (classe *Bad*, representando 30% da base) para minimizar perdas financeiras com a concessão indevida de crédito.
* **Desafio Técnico:** Evoluir de um modelo linear de referência (*baseline*) para um modelo não-linear e calibrar o limiar de decisão (*Threshold Tuning*) focado na precisão da classe crítica.
* **Evolução da Performance (Precisão para a Classe Bad):**
  * **Baseline (Regressão Logística - Limiar 0.50):** `30.77%` *(desempenho próximo à proporção natural da base)*.
  * **Seleção do Algoritmo (XGBoost Otimizado - Limiar 0.50):** `38.81%` *(ganho de +8.04% ao capturar padrões não-lineares)*.
  * **Modelo Final (XGBoost Otimizado - Limiar Ajustado 0.65):** `45.95%` *(ganho total de +15.18% sobre o baseline ao aplicar o Threshold Tuning)*.
* **Resultado para o Negócio:** Redução acentuada na aprovação de clientes inadimplentes (*Falsos Positivos*), garantindo maior proteção ao capital e permitindo o ajuste fino da esteira de crédito conforme o cenário econômico.
* **Tecnologias:** `Python`, `Scikit-Learn`, `XGBoost`, `Hyperparameter Tuning`, `Threshold Tuning`.

---

## 🛠️ Tecnologias e Metodologia Utilizadas

Ambos os projetos foram desenvolvidos utilizando boas práticas de Engenharia de Machine Learning:

* **Pipelines do Scikit-Learn:** Garantia contra *data leakage* (vazamento de dados) entre conjuntos de treino e teste.
* **Tratamento Customizado de Dados:** Criação de classes para remoção e substituição de *outliers* via amplitude interquartil (IQR).
* **Ajustes de Distribuição e Escala:** Aplicação de `PowerTransformer` (Box-Cox/Yeo-Johnson) para normalização e `StandardScaler` para padronização.
* **Parcimônia de Modelos:** Priorização da Navalha de Ockham — entregando modelos leves, interpretáveis, fáceis de monitorar em produção e com performance equivalente aos modelos complexos.



