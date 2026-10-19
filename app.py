import json
import os

import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
model = joblib.load(os.path.join(BASE, "model_jurnal.pkl"))

# Hasil jurnal (Sidiq dkk., JATI 2024), dari confusion matrix pada Gambar 7 jurnal
JURNAL = {
    "algoritma": "Decision Tree C4.5 (RapidMiner)",
    "n_test": 818,
    "accuracy": 0.9364,
    "precision": 0.125,
    "recall": 0.05,
    "tn": 764, "fp": 14, "fn": 38, "tp": 2,
}


@app.route("/evaluasi")
def evaluasi():
    with open(os.path.join(BASE, "metrics.json")) as f:
        m = json.load(f)
    return render_template("evaluasi.html", m=m, j=JURNAL)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    form_values = {
        "gender": request.form.get("gender", "").strip().capitalize(),
        "age": request.form.get("age", "").strip(),
        "hypertension": request.form.get("hypertension", "").strip(),
        "ever_married": request.form.get("ever_married", "").strip().capitalize(),
        "work_type": request.form.get("work_type", "").strip(),
        "Residence_type": request.form.get("Residence_type", "").strip().capitalize(),
        "avg_glucose_level": request.form.get("avg_glucose_level", "").strip(),
        "bmi": request.form.get("bmi", "").strip(),
        "smoking_status": request.form.get("smoking_status", "").strip(),
    }

    errors = {}
    for key, label in {
        "gender": "Jenis kelamin",
        "age": "Usia",
        "hypertension": "Hipertensi",
        "ever_married": "Status menikah",
        "work_type": "Jenis pekerjaan",
        "Residence_type": "Tipe tempat tinggal",
        "avg_glucose_level": "Kadar glukosa",
        "smoking_status": "Status merokok",
    }.items():
        if not form_values[key]:
            errors[key] = f"{label} tidak boleh kosong"

    if errors:
        return render_template("index.html", errors=errors, values=form_values)

    try:
        data = {
            "gender": form_values["gender"],
            "age": float(form_values["age"].replace(",", ".")),
            "hypertension": str(int(form_values["hypertension"])),
            "ever_married": form_values["ever_married"],
            "work_type": form_values["work_type"],
            "Residence_type": form_values["Residence_type"],
            "avg_glucose_level": float(form_values["avg_glucose_level"].replace(",", ".")),
            "bmi": float(form_values["bmi"].replace(",", ".")) if form_values["bmi"] else float("nan"),
            "smoking_status": form_values["smoking_status"],
        }
    except ValueError:
        return render_template("index.html", errors={"form": "Pastikan semua angka diisi dengan benar."}, values=form_values)

    df_input = pd.DataFrame([data])

    prediction = int(model.predict(df_input)[0])
    probability = model.predict_proba(df_input)[0]

    result = {
        "label": "Stroke" if prediction == 1 else "Tidak Stroke",
        "prediction": prediction,
        "probability": {
            "Tidak Stroke": round(float(probability[0]) * 100, 2),
            "Stroke": round(float(probability[1]) * 100, 2),
        },
    }

    return render_template("index.html", result=result, values=form_values)


if __name__ == "__main__":
    app.run(debug=True)