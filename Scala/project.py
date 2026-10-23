
import pandas as pd
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, trim
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

spark = SparkSession.builder.appName("ChurnProject").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

CSV_PATH = r"D:\Downloads Browser\telco-customer-churn-by-IBM.csv"

data = spark.read.csv(CSV_PATH, header=True, inferSchema=True)

total_rows = data.count()
total_columns = len(data.columns)

churn_data = data.groupBy("Churn").count().toPandas()

data = data.withColumn(
    "TotalCharges",
    when(trim(col("TotalCharges")) == "", None)
    .otherwise(col("TotalCharges").cast("double"))
)

data = data.na.drop()

feature_columns = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features"
)

data_prepared = assembler.transform(data).select("features", "Churn")

indexer = StringIndexer(inputCol="Churn", outputCol="label")
data_prepared = indexer.fit(data_prepared).transform(data_prepared)

train_data, test_data = data_prepared.randomSplit([0.8, 0.2], seed=42)

train_count = train_data.count()
test_count = test_data.count()

lr = LogisticRegression(featuresCol="features", labelCol="label")
lr_model = lr.fit(train_data)

lr_predictions = lr_model.transform(test_data)

evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

auc = evaluator.evaluate(lr_predictions)

correct = lr_predictions.filter(
    col("label") == col("prediction")
).count()

total = lr_predictions.count()
lr_accuracy = (correct / total) * 100

lr_cm_data = lr_predictions.groupBy(
    "label", "prediction"
).count().collect()

lr_cm_counts = {
    (r["label"], r["prediction"]): r["count"]
    for r in lr_cm_data
}

lr_cm = [
    [lr_cm_counts.get((0.0, 0.0), 0), lr_cm_counts.get((0.0, 1.0), 0)],
    [lr_cm_counts.get((1.0, 0.0), 0), lr_cm_counts.get((1.0, 1.0), 0)]
]

dt = DecisionTreeClassifier(
    featuresCol="features",
    labelCol="label"
)

dt_model = dt.fit(train_data)
dt_predictions = dt_model.transform(test_data)

dt_correct = dt_predictions.filter(
    col("label") == col("prediction")
).count()

dt_total = dt_predictions.count()
dt_accuracy = (dt_correct / dt_total) * 100

dt_cm_data = dt_predictions.groupBy(
    "label", "prediction"
).count().collect()

dt_cm_counts = {
    (r["label"], r["prediction"]): r["count"]
    for r in dt_cm_data
}

dt_cm = [
    [dt_cm_counts.get((0.0, 0.0), 0), dt_cm_counts.get((0.0, 1.0), 0)],
    [dt_cm_counts.get((1.0, 0.0), 0), dt_cm_counts.get((1.0, 1.0), 0)]
]

results = pd.DataFrame(
    {
        "Accuracy": [lr_accuracy, dt_accuracy],
        "AUC": [auc, None]
    },
    index=["Logistic Regression", "Decision Tree"]
)

fig, axes = plt.subplots(2, 2, figsize=(15, 11))

fig.suptitle("Customer Churn Prediction using PySpark MLlib",
             fontsize=19, fontweight="bold", y=0.99)

fig.text(
    0.5, 0.925,
    f"Total Rows: {total_rows}     Total Columns: {total_columns}     "
    f"Training Data: {train_count}     Testing Data: {test_count}",
    ha="center",
    fontsize=11
)   # keep your f-string, only change 0.92 to 0.925

ax = axes[0][0]

ax.bar(["Logistic Regression", "Decision Tree"],
       [lr_accuracy, dt_accuracy],
       color=["#2a9d8f", "#e9c46a"])

ax.set_title("Model Accuracy Comparison")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(65, 85)

for i, value in enumerate([lr_accuracy, dt_accuracy]):
    ax.text(i, value + 1, f"{value:.2f}%", ha="center")

ax = axes[0][1]

ax.bar(churn_data["Churn"], churn_data["count"],
       color=["#2a9d8f", "#e76f51"])

ax.set_title("Churn Distribution")
ax.set_xlabel("Churn")
ax.set_ylabel("Number of Customers")

for i, value in enumerate(churn_data["count"]):
    ax.text(i, value + 50, str(value), ha="center")

ax = axes[1][0]

ax.imshow(lr_cm, cmap="Blues", aspect="auto")     # and the same for dt_cm
ax.set_title("Logistic Regression Confusion Matrix")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Predicted No", "Predicted Yes"])
ax.set_yticks([0, 1])
ax.set_yticklabels(["Actual No", "Actual Yes"])
ax.set_xlabel("Prediction")
ax.set_ylabel("Actual")

for i in range(2):
    for j in range(2):
        ax.text(
            j, i, lr_cm[i][j],
            ha="center",
            va="center",
            fontsize=16
        )

ax = axes[1][1]

ax.imshow(dt_cm, cmap="Greens")
ax.set_title("Decision Tree Confusion Matrix")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Predicted No", "Predicted Yes"])
ax.set_yticks([0, 1])
ax.set_yticklabels(["Actual No", "Actual Yes"])
ax.set_xlabel("Prediction")
ax.set_ylabel("Actual")

for i in range(2):
    for j in range(2):
        ax.text(
            j, i, dt_cm[i][j],
            ha="center",
            va="center",
            fontsize=16
        )

plt.tight_layout(rect=[0, 0.02, 1, 0.88], h_pad=4)

plt.savefig("churn_dashboard.png", dpi=150)

plt.show()

spark.stop()
