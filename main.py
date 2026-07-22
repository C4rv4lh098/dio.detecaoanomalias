"""
Detecção de Fraudes em Transações Bancárias
============================================

Arquivo educacional organizado em blocos.
Cada linha da base representa uma transação e a coluna Class indica:
0 = transação normal
1 = fraude
"""

# ============================================================
# BLOCO 1 — IMPORTAÇÃO E CARREGAMENTO DOS DADOS
# ============================================================
"""
Usamos pandas para carregar e manipular dados tabulares.
O df.head() permite verificar rapidamente se a base foi lida corretamente.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)

print("\nPrimeiras linhas da base:")
print(df.head())


# ============================================================
# BLOCO 2 — ANÁLISE DO DESBALANCEAMENTO
# ============================================================
"""
Fraudes são raras. Um modelo pode ter alta acurácia apenas prevendo quase
sempre a classe 0. Por isso, verificamos a proporção das classes antes de
qualquer treinamento.
"""

print("\nProporção das classes:")
print(df["Class"].value_counts(normalize=True))


# ============================================================
# BLOCO 3 — ENGENHARIA DE ATRIBUTOS
# ============================================================
"""
A variável Amount pode ser muito assimétrica: muitas transações pequenas e
poucas muito altas. A transformação logarítmica reduz essa assimetria e o
impacto de valores extremos.

A padronização será feita depois, dentro de um Pipeline, para evitar vazamento
de informação entre treino e teste.
"""

df["Amount_log"] = np.log1p(df["Amount"])


# ============================================================
# BLOCO 4 — SEPARAÇÃO ENTRE VARIÁVEIS E ALVO
# ============================================================
"""
X contém as variáveis de entrada e y contém a resposta que desejamos prever.
Removemos Amount porque já criamos Amount_log, evitando duas versões muito
parecidas da mesma informação.
"""

X = df.drop(columns=["Class", "Amount"])
y = df["Class"]


# ============================================================
# BLOCO 5 — DIVISÃO ENTRE TREINO E TESTE
# ============================================================
"""
O conjunto de treino ensina o modelo. O conjunto de teste mede o desempenho em
dados não vistos. O stratify mantém a proporção de fraudes nas duas partes.
"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    stratify=y,
    test_size=0.30,
    random_state=42
)


# ============================================================
# BLOCO 6 — REGRESSÃO LOGÍSTICA COM PIPELINE
# ============================================================
"""
A Regressão Logística é um bom modelo inicial: simples, rápida e interpretável.
O Pipeline garante que a padronização seja aprendida apenas com os dados de
treino e aplicada corretamente ao teste.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
y_probs = pipeline.predict_proba(X_test)[:, 1]


# ============================================================
# BLOCO 7 — RELATÓRIO DE CLASSIFICAÇÃO
# ============================================================
"""
Acurácia isolada é enganosa em bases desbalanceadas.

Precision: das previsões de fraude, quantas eram realmente fraude?
Recall: das fraudes existentes, quantas foram encontradas?
F1-score: equilíbrio entre precision e recall.
Support: quantidade de exemplos de cada classe.
"""

from sklearn.metrics import classification_report

print("\nRegressão Logística — relatório de classificação:")
print(classification_report(y_test, y_pred))


# ============================================================
# BLOCO 8 — CURVA ROC E AUC
# ============================================================
"""
A curva ROC compara a taxa de verdadeiros positivos com a taxa de falsos
positivos para vários limiares. A AUC resume a capacidade de separar as classes.
Em fraude, deve ser analisada junto da curva Precision-Recall.
"""

from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, _ = roc_curve(y_test, y_probs)

plt.figure()
plt.plot(fpr, tpr)
plt.title("Curva ROC — Regressão Logística")
plt.xlabel("Taxa de falsos positivos")
plt.ylabel("Taxa de verdadeiros positivos")
plt.tight_layout()
plt.show()

print("AUC:", roc_auc_score(y_test, y_probs))


# ============================================================
# BLOCO 9 — CURVA PRECISION-RECALL
# ============================================================
"""
Essa curva é especialmente útil quando a classe positiva é rara.
Recall alto encontra mais fraudes; precision alta reduz alarmes falsos.
"""

from sklearn.metrics import precision_recall_curve

precision, recall, _ = precision_recall_curve(y_test, y_probs)

plt.figure()
plt.plot(recall, precision)
plt.title("Curva Precision-Recall — Regressão Logística")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.tight_layout()
plt.show()


# ============================================================
# BLOCO 10 — AJUSTE DO LIMIAR DE DECISÃO
# ============================================================
"""
O limiar padrão costuma ser 0,5. Ao reduzir para 0,3, o modelo tende a detectar
mais fraudes, aumentando o recall, mas também pode gerar mais falsos positivos.
A escolha depende do custo de deixar uma fraude passar e do custo de bloquear
uma transação legítima.
"""

threshold = 0.30
y_pred_custom = (y_probs > threshold).astype(int)

print("\nRegressão Logística — threshold de 0.30:")
print(classification_report(y_test, y_pred_custom))


# ============================================================
# BLOCO 11 — UNDERSAMPLING
# ============================================================
"""
O undersampling reduz a classe majoritária até ficar do tamanho da classe de
fraude. É rápido, mas descarta muitas transações normais e pode perder informação.
O balanceamento é feito somente no treino; o teste mantém a distribuição real.
"""

fraudes_train = X_train[y_train == 1].copy()
normais_train = X_train[y_train == 0].sample(
    n=len(fraudes_train),
    random_state=42
)

X_train_under = pd.concat([fraudes_train, normais_train])
y_train_under = y_train.loc[X_train_under.index]

under_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, random_state=42))
])

under_pipeline.fit(X_train_under, y_train_under)
y_pred_under = under_pipeline.predict(X_test)

print("\nRegressão Logística com undersampling:")
print(classification_report(y_test, y_pred_under))


# ============================================================
# BLOCO 12 — OVERSAMPLING COM SMOTE
# ============================================================
"""
O SMOTE cria exemplos sintéticos da classe minoritária, ajudando o modelo a
aprender padrões de fraude sem simplesmente duplicar registros.
Ele deve ser aplicado apenas no treino para evitar vazamento de dados.

Instalação, se necessário:
pip install imbalanced-learn
"""

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    smote_pipeline = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("model", LogisticRegression(max_iter=1000, random_state=42))
    ])

    smote_pipeline.fit(X_train, y_train)
    y_pred_smote = smote_pipeline.predict(X_test)

    print("\nRegressão Logística com SMOTE:")
    print(classification_report(y_test, y_pred_smote))

except ImportError:
    print("\nSMOTE não executado. Instale com: pip install imbalanced-learn")


# ============================================================
# BLOCO 13 — RANDOM FOREST
# ============================================================
"""
Random Forest combina várias árvores e captura relações não lineares.
class_weight='balanced' aumenta o peso da classe de fraude, ajudando o modelo a
não ignorá-la. Esse modelo não exige padronização das variáveis.
"""

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\nRandom Forest — relatório de classificação:")
print(classification_report(y_test, y_pred_rf))


# ============================================================
# BLOCO 14 — XGBOOST
# ============================================================
"""
O XGBoost cria árvores sequencialmente, cada uma corrigindo erros das anteriores.
Costuma ter ótimo desempenho em dados tabulares.

scale_pos_weight é calculado pela razão entre transações normais e fraudes,
refletindo o desbalanceamento real do treino.

Instalação, se necessário:
pip install xgboost
"""

try:
    from xgboost import XGBClassifier

    quantidade_normais = (y_train == 0).sum()
    quantidade_fraudes = (y_train == 1).sum()
    scale_pos_weight = quantidade_normais / quantidade_fraudes

    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)

    print("\nXGBoost — relatório de classificação:")
    print(classification_report(y_test, y_pred_xgb))

except ImportError:
    xgb = None
    print("\nXGBoost não executado. Instale com: pip install xgboost")


# ============================================================
# BLOCO 15 — IMPORTÂNCIA DAS VARIÁVEIS
# ============================================================
"""
A importância mostra quais variáveis foram mais usadas pelo modelo.
Ela ajuda na interpretação, mas não prova que uma variável causa fraude.
"""

if xgb is not None:
    importancias = pd.Series(
        xgb.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)

    plt.figure(figsize=(12, 5))
    importancias.head(15).plot(kind="bar")
    plt.title("15 variáveis mais importantes — XGBoost")
    plt.xlabel("Variável")
    plt.ylabel("Importância")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


# ============================================================
# BLOCO 16 — AJUSTE DE HIPERPARÂMETROS
# ============================================================
"""
O GridSearchCV testa combinações de hiperparâmetros com validação cruzada.
Aqui usamos recall como critério, pois encontrar fraudes é prioridade.
Porém, recall sozinho pode aumentar falsos positivos; precision, F1 e PR-AUC
também devem ser observados.
"""

if xgb is not None:
    from sklearn.model_selection import GridSearchCV

    param_grid = {
        "max_depth": [3, 5],
        "n_estimators": [50, 100]
    }

    grid = GridSearchCV(
        estimator=XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        ),
        param_grid=param_grid,
        scoring="recall",
        cv=3,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("\nMelhores hiperparâmetros:")
    print(grid.best_params_)


# ============================================================
# BLOCO 17 — EXPLICABILIDADE COM SHAP
# ============================================================
"""
O SHAP explica quanto cada variável contribuiu para aumentar ou reduzir a
probabilidade de fraude. Isso é valioso para auditoria e investigação.

Instalação, se necessário:
pip install shap
"""

if xgb is not None:
    try:
        import shap

        amostra_shap = X_test.iloc[:100].copy()
        explainer = shap.Explainer(xgb)
        shap_values = explainer(amostra_shap)
        shap.plots.bar(shap_values)

    except ImportError:
        print("\nSHAP não executado. Instale com: pip install shap")
