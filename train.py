import pandas as pd
import json
import mlflow
from mlflow import sklearn as mlflow_sklearn
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
import joblib
import os
import time
import dagshub
from mlflow.exceptions import MlflowException

os.environ["MLFLOW_TRACKING_USERNAME"] = "2022bcs0060-arjunsreenivas"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "40f9121076a8b3b6fb330ef8309e821103b4d5d7"

mlflow.set_tracking_uri("https://dagshub.com/2022bcs0060-arjunsreenivas/2022bcs0060-assignment2.mlflow")
mlflow.set_experiment("churn_prediction")

# -------------------------
# DagsHub Authentication
# -------------------------


with mlflow.start_run() as run:
    data = pd.read_csv("dataset/telco_customer_churn.csv")
    X = data.drop("risk", axis=1)
    y = data["risk"]
    mlflow.log_param("dataset", "telco_customer_churn.csv")
    mlflow.log_param("features", list(X.columns))

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    # Pipeline
    n_estimators = 100
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=n_estimators, random_state=42))
    ])

    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("dataset size",len(data))

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    f1 = f1_score(y_test, y_pred, average="weighted")
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']

    mlflow.log_metric("f1_score", float(f1))
    mlflow.log_metric("precision", float(precision))
    mlflow.log_metric("recall", float(recall))

    # Save locally + log artifacts
    metrics = {"f1_score": f1, "classification_report": report}
    os.makedirs("output", exist_ok=True)
    with open("output/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    joblib.dump(pipeline, "output/model.pkl")
    mlflow.log_artifacts("output")

    mlflow_sklearn.log_model(pipeline, name="model")

    print("Training complete with DagsHub tracking.")
    run_id = run.info.run_id

model_uri = f"runs:/{run_id}/model"

client = MlflowClient()

try:
    client.create_registered_model("assignment2-churn")
except Exception:
    pass 

artifact_uri = f"{client.get_run(run_id).info.artifact_uri}/model"

max_retries = 10
for attempt in range(max_retries):
    try:
        mv = client.create_model_version(
            name="assignment2-churn",
            source=artifact_uri,
            run_id=run_id,
        )
        print(f"Registered version: {mv.version}")
        break
    except Exception as e:
        if attempt < max_retries - 1:
            print(f"Retrying ({attempt + 1}/{max_retries})... {e}")
            time.sleep(5)
        else:
            raise