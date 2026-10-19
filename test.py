from pathlib import Path

import joblib
import pandas as pd

# model = joblib.load("model_mlp_oversampling.pkl")
# model = joblib.load("model_mlp.pkl")
model = joblib.load("model_jurnal.pkl")

print("=" * 60)
print("PREDIKSI STROKE MENGGUNAKAN MLP")
print("=" * 60)
print()

print("Masukkan data pasien")
print("-" * 60)

gender = input("Jenis kelamin (Male/Female)         : ").capitalize()
age = float(input("Usia                             : "))
hypertension = input("Hipertensi (0=Tidak, 1=Ya)    : ")
heart_disease = input("Penyakit jantung (0, 1)      : ")
ever_married = input("Pernah menikah (Yes/No)       : ").capitalize()
work_type = input("Jenis pekerjaan (Private/Self-employed/Govt_job/children/Never_worked): ")
residence_type = input("Tempat tinggal (Urban/Rural)      : ").capitalize()
avg_glucose_level = float(input("Rata-rata kadar glukosa  : "))
bmi_input = input("BMI (kosongkan jika tidak diketahui)   : ")
smoking_status = input("Status merokok (never smoked/formerly smoked/smokes/Unknown): ")

data_input = pd.DataFrame([{
    "gender": gender,
    "age": age,
    "hypertension": hypertension,
    "heart_disease": heart_disease,
    "ever_married": ever_married,
    "work_type": work_type,
    "Residence_type": residence_type,
    "avg_glucose_level": avg_glucose_level,
    "bmi": float(bmi_input) if bmi_input.strip() else float("nan"),
    "smoking_status": smoking_status,
}])

prediction = int(model.predict(data_input)[0])
probability = model.predict_proba(data_input)[0]

print()
print("=" * 60)
print("HASIL PREDIKSI")
print("=" * 60)
print("Stroke      :", prediction)
print(
    "Keterangan  :",
    "Stroke" if prediction == 1 else "Tidak Stroke",
)
print(f"Probabilitas Tidak Stroke : {probability[0] * 100:.2f}%")
print(f"Probabilitas Stroke       : {probability[1] * 100:.2f}%")
print("=" * 60)
