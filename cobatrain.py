import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

file_path = "healthcare-dataset-stroke-data.csv"
df = pd.read_csv(file_path)

df = df.drop(columns=["id"])
df = df[df["gender"] != "Other"].copy()

X = df.drop("stroke", axis=1)
y = df["stroke"]

numeric_features = ["age", "avg_glucose_level", "bmi"]
categorical_features = [
    "gender",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status",
]

X[["hypertension", "heart_disease"]] = X[["hypertension", "heart_disease"]].astype(str)

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

train_data = pd.concat([X_train, y_train], axis=1)
mayor = train_data[train_data["stroke"] == 0]
minor = train_data[train_data["stroke"] == 1]
minor_up = resample(minor, replace=True, n_samples=len(mayor), random_state=42)
train_balanced = pd.concat([mayor, minor_up]).sample(frac=1, random_state=42)

X_train_bal = train_balanced.drop("stroke", axis=1)
y_train_bal = train_balanced["stroke"]

mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    alpha=0.0001,
    learning_rate_init=0.001,
    max_iter=300,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("mlp", mlp),
    ]
)

model.fit(X_train_bal, y_train_bal)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

classification_report(y_test, y_pred, target_names=["Tidak Stroke", "Stroke"])
confusion_matrix(y_test, y_pred)

joblib.dump(model, "model_mlp.pkl")