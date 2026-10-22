import os
import sys
from itertools import chain

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import (LogisticRegression,
                                       DecisionTreeClassifier,
                                       RandomForestClassifier)
from pyspark.ml.evaluation import (MulticlassClassificationEvaluator,
                                   BinaryClassificationEvaluator)

spark = (SparkSession.builder
         .appName("ChurnProject")
         .master("local[*]")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

CSV_PATH = r"D:\Downloads Browser\telco-customer-churn-by-IBM.csv"

data = spark.read.csv(CSV_PATH, header=True, inferSchema=True)
print("Loaded:", data.count(), "rows,", len(data.columns), "columns")

data = data.withColumn(
    "TotalCharges",
    F.when(F.trim(F.col("TotalCharges")) == "", None)
     .otherwise(F.col("TotalCharges").cast("double"))
)
data = data.dropna(subset=["TotalCharges"])   
data = data.drop("customerID")               
print("After cleaning:", data.count(), "rows,", len(data.columns), "columns")

data.groupBy("Churn").count().show()

cat_cols = [c for c, t in data.dtypes if t == "string" and c != "Churn"]
num_cols = [c for c, t in data.dtypes if t in ("int", "double")]

label_indexer = StringIndexer(inputCol="Churn", outputCol="label")
indexers = [StringIndexer(inputCol=c, outputCol=c + "_idx") for c in cat_cols]
encoder = OneHotEncoder(inputCols=[c + "_idx" for c in cat_cols],
                        outputCols=[c + "_vec" for c in cat_cols])
assembler = VectorAssembler(inputCols=num_cols + [c + "_vec" for c in cat_cols],
                            outputCol="features")

prep_stages = [label_indexer] + indexers + [encoder, assembler]

train, test = data.randomSplit([0.8, 0.2], seed=42)
print("Train:", train.count(), " Test:", test.count())


def evaluate(pred, name):
    mc = MulticlassClassificationEvaluator(labelCol="label",
                                           predictionCol="prediction")
    acc = mc.setMetricName("accuracy").evaluate(pred)
    prec = mc.setMetricName("weightedPrecision").evaluate(pred)
    rec = mc.setMetricName("weightedRecall").evaluate(pred)
    f1 = mc.setMetricName("f1").evaluate(pred)
    auc = BinaryClassificationEvaluator(labelCol="label",
                                        metricName="areaUnderROC").evaluate(pred)
    print(f"{name}: Accuracy={acc:.3f} Precision={prec:.3f} "
          f"Recall={rec:.3f} F1={f1:.3f} AUC={auc:.3f}")
    return acc, prec, rec, f1, auc


lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=100)
lr_model = Pipeline(stages=prep_stages + [lr]).fit(train)
lr_pred = lr_model.transform(test)
lr_scores = evaluate(lr_pred, "Logistic Regression")

dt = DecisionTreeClassifier(featuresCol="features", labelCol="label",
                            maxDepth=5, seed=42)
dt_model = Pipeline(stages=prep_stages + [dt]).fit(train)
dt_pred = dt_model.transform(test)
dt_scores = evaluate(dt_pred, "Decision Tree")

rf = RandomForestClassifier(featuresCol="features", labelCol="label",
                            numTrees=100, seed=42)
rf_model = Pipeline(stages=prep_stages + [rf]).fit(train)
rf_pred = rf_model.transform(test)
rf_scores = evaluate(rf_pred, "Random Forest")

results = pd.DataFrame(
    [lr_scores, dt_scores, rf_scores],
    columns=["Accuracy", "Precision", "Recall", "F1", "AUC"],
    index=["Logistic Regression", "Decision Tree", "Random Forest"],
).round(3)
print("\n", results, "\n")


cm_counts = {(r["label"], r["prediction"]): r["count"]
             for r in lr_pred.groupBy("label", "prediction").count().collect()}
cm = [[cm_counts.get((0.0, 0.0), 0), cm_counts.get((0.0, 1.0), 0)],
      [cm_counts.get((1.0, 0.0), 0), cm_counts.get((1.0, 1.0), 0)]]
total_test = sum(sum(row) for row in cm)
correct = cm[0][0] + cm[1][1]
churn_recall = 100 * cm[1][1] / (cm[1][0] + cm[1][1])
print("Confusion matrix:", cm)


contract_counts = {}
for r in data.groupBy("Contract", "Churn").count().collect():
    contract_counts.setdefault(r["Contract"], {})[r["Churn"]] = r["count"]
contracts = ["Month-to-month", "One year", "Two year"]
contracts = [c for c in contracts if c in contract_counts]
rates = [100 * contract_counts[c].get("Yes", 0) /
         (contract_counts[c].get("Yes", 0) + contract_counts[c].get("No", 0))
         for c in contracts]


attrs = rf_pred.schema["features"].metadata["ml_attr"]["attrs"]
names = [a["name"] for a in chain(*attrs.values())]
importances = rf_model.stages[-1].featureImportances.toArray()
top = sorted(zip(names, importances), key=lambda x: x[1], reverse=True)[:10]
top_labels = [t[0] for t in top][::-1]     # reversed so biggest bar is on top
top_values = [t[1] for t in top][::-1]
print("Top features:", [t[0] for t in top[:5]])


fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("Customer Churn Prediction using PySpark MLlib",
             fontsize=19, fontweight="bold")

objectives = (
    "Objectives:  (1) Clean and prepare telecom data in Spark    "
    "(2) Build a feature pipeline: StringIndexer, OneHotEncoder, VectorAssembler\n"
    "(3) Train 3 MLlib models: Logistic Regression, Decision Tree, Random Forest    "
    "(4) Compare models and find why customers churn"
)
fig.text(0.5, 0.925, objectives, ha="center", va="center", fontsize=10)


ax = axes[0][0]
ax.bar(results.index, results["AUC"], color=["#2a9d8f", "#e9c46a", "#457b9d"])
ax.set_title("Model comparison (AUC, higher is better)")
ax.set_ylim(0.6, 0.9)
for i, v in enumerate(results["AUC"]):
    ax.text(i, v + 0.004, f"{v:.3f}", ha="center", fontweight="bold")

ax = axes[0][1]
ax.imshow(cm, cmap="Blues")
ax.set_title(f"Confusion matrix: Logistic Regression\n"
             f"{correct} of {total_test} correct ({100 * correct / total_test:.1f}%), "
             f"churn recall {churn_recall:.0f}%")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Predicted stay", "Predicted churn"])
ax.set_yticks([0, 1])
ax.set_yticklabels(["Actually stayed", "Actually churned"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=18,
                color="white" if cm[i][j] > max(max(cm)) / 2 else "black")


ax = axes[1][0]
ax.bar(contracts, rates, color="#e76f51")
ax.set_title("Churn rate by contract type (%)")c
ax.set_ylim(0, max(rates) + 8)
for i, v in enumerate(rates):
    ax.text(i, v + 0.8, f"{v:.1f}%", ha="center", fontweight="bold")

ax = axes[1][1]
ax.barh(top_labels, top_values, color="#6a4c93")
ax.set_title("Top 10 features driving churn (Random Forest)")
ax.set_xlabel("Importance")

plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig("churn_dashboard.png", dpi=150)
print("Saved churn_dashboard.png in:", os.getcwd())
plt.show()

spark.stop()