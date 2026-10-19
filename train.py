import pandas as pd
import numpy as np
import joblib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix

df = pd.read_csv("healthcare-dataset-stroke-data.csv")

print("=" * 60)
print("DATASET")
print("=" * 60)
print("Jumlah data :", len(df))
print("Jumlah kolom:", len(df.columns))
print()
print(df.head())
print()

print("=" * 60)
print("INFORMASI DATA")
print("=" * 60)
print(df.info())
print()

print("Missing value:")
print(df.isnull().sum())
print()

print("Distribusi target (stroke):")
print(df["stroke"].value_counts())
print()

n_rows = len(df)
n_cols = len(df.columns)
n_stroke = int((df["stroke"] == 1).sum())
n_nostroke = int((df["stroke"] == 0).sum())
n_missing_bmi = int(df["bmi"].isnull().sum())

df = df.drop(columns=["id", "heart_disease"])

print("Jumlah data setelah cleaning:", len(df))
print()

X = df.drop("stroke", axis=1)
y = df["stroke"]

numeric_features = [
    "age",
    "avg_glucose_level",
    "bmi"
]

categorical_features = [
    "gender",
    "hypertension",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status"
]

X["hypertension"] = X["hypertension"].astype(str)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("=" * 60)
print("PEMBAGIAN DATA")
print("=" * 60)
print("Data training:", X_train.shape)
print("Data testing :", X_test.shape)
print()

print("Distribusi target data training:")
print(y_train.value_counts())
print()

dt = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=10,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("dt", dt)
    ]
)

print("=" * 60)
print("TRAINING DECISION TREE")
print("=" * 60)

model.fit(X_train, y_train)

tree = model.named_steps["dt"]

print("Training selesai.")
print("Kedalaman pohon:", tree.get_depth())
print("Jumlah daun    :", tree.get_n_leaves())
print()

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print("=" * 60)
print("HASIL EVALUASI")
print("=" * 60)
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}  (kelas Stroke)")
print(f"Recall   : {recall:.4f}  (kelas Stroke)")
print()

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Tidak Stroke", "Stroke"], zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")
print()

feature_names = model.named_steps["preprocessor"].get_feature_names_out()

plt.figure(figsize=(22, 10))
plot_tree(
    tree,
    feature_names=feature_names,
    class_names=["Tidak Stroke", "Stroke"],
    max_depth=3,
    filled=True,
    rounded=True,
    fontsize=8
)
plt.tight_layout()
plt.savefig("decision_tree.png", dpi=120)
plt.close()

print("Gambar pohon disimpan: decision_tree.png")

joblib.dump(model, "model_jurnal.pkl")

with open("metrics.json", "w") as f:
    json.dump({
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_stroke": n_stroke,
        "n_nostroke": n_nostroke,
        "n_missing_bmi": n_missing_bmi,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "depth": int(tree.get_depth()),
        "leaves": int(tree.get_n_leaves()),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
    }, f, indent=2)