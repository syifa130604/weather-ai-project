from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import os

# Mengamankan import tensorflow agar jika env bermasalah, aplikasi tidak langsung mati
try:
    from tensorflow.keras.models import load_model
except ImportError:
    load_model = None

# =========================================
# INITIALIZATION
# =========================================
app = Flask(__name__)

# Mengatur path folder agar fleksibel
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models')
DOCS_PATH = os.path.join(BASE_DIR, '..', 'docs')

# Inisialisasi variabel model sebagai None terlebih dahulu
lr_model = ann_model = lstm_model = backprop_model = kmeans_model = scaler = None

# =========================================
# LOAD MODELS & SCALER
# =========================================
try:
    # Memuat 5 Algoritma: ANN, Backpro, LSTM (RNN), Linear Regression, K-Means
    if os.path.exists(os.path.join(MODEL_PATH, 'linear_regression.pkl')):
        lr_model = joblib.load(os.path.join(MODEL_PATH, 'linear_regression.pkl'))
        
    if load_model is not None:
        if os.path.exists(os.path.join(MODEL_PATH, 'ann_model.h5')):
            ann_model = load_model(os.path.join(MODEL_PATH, 'ann_model.h5'), compile=False)
        if os.path.exists(os.path.join(MODEL_PATH, 'lstm_model.h5')):
            lstm_model = load_model(os.path.join(MODEL_PATH, 'lstm_model.h5'), compile=False)
        if os.path.exists(os.path.join(MODEL_PATH, 'backpropagation_model.h5')):
            backprop_model = load_model(os.path.join(MODEL_PATH, 'backpropagation_model.h5'), compile=False)
            
    if os.path.exists(os.path.join(MODEL_PATH, 'kmeans_model.pkl')):
        kmeans_model = joblib.load(os.path.join(MODEL_PATH, 'kmeans_model.pkl'))
        
    if os.path.exists(os.path.join(MODEL_PATH, 'scaler.pkl')):
        scaler = joblib.load(os.path.join(MODEL_PATH, 'scaler.pkl'))
except Exception as e:
    print(f"Error Loading Models: {e}")

# =========================================
# ROUTES - NAVIGATION
# =========================================

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/visualization')
def visualization():
    return render_template('visualization.html')

# =========================================
# ROUTE - PREDICTION PAGE & LOGIC (TERPADU)
# =========================================

@app.route('/predict-page', methods=['GET', 'POST'])
def predict_page():
    # JIKA USER BARU MEMBUKA HALAMAN (GET)
    if request.method == 'GET':
        return render_template('predict.html', weather_status=None)
        
    # JIKA USER MENEKAN TOMBOL ANALISIS (POST)
    try:
        # 1. Pastikan komponen dasar paling krusial (Scaler) ter-load
        if scaler is None:
            return render_template(
                'predict.html', 
                weather_status="⚠️ Gagal: Komponen pre-processing 'scaler.pkl' tidak ditemukan atau gagal dimuat di server.",
                lr_prediction='error', ann_prediction=0, lstm_prediction=0, backprop_prediction=0, kmeans_prediction=0,
                best_model="System Pre-check"
            )

        # 2. Mengambil input dari form secara aman
        inputs = [
            float(request.form.get('temp_avg', 0)),
            float(request.form.get('temp_max', 0)),
            float(request.form.get('temp_min', 0)),
            float(request.form.get('press', 0)),
            float(request.form.get('humid_avg', 0)),
            float(request.form.get('ws_avg', 0)),
            float(request.form.get('max_ws', 0)),
            float(request.form.get('light_hour', 0))
        ]
        
        # 3. Preprocessing & Scaling
        data_array = np.array([inputs])
        scaled_data = scaler.transform(data_array)

        # 4. Eksekusi Prediksi dengan Proteksi Fallback Mandiri
        results = {}
        
        # Linear Regression
        if lr_model is not None:
            lr_p = float(lr_model.predict(scaled_data)[0])
            results["Regresi Linear"] = lr_p
        else:
            lr_p = 0.0

        # ANN
        if ann_model is not None:
            ann_p = float(ann_model.predict(scaled_data)[0][0])
            results["ANN (Neural Network)"] = ann_p
        else:
            ann_p = 0.0

        # LSTM
        if lstm_model is not None:
            lstm_in = scaled_data.reshape((scaled_data.shape[0], 1, scaled_data.shape[1]))
            lstm_p = float(lstm_model.predict(lstm_in)[0][0])
            results["LSTM (RNN)"] = lstm_p
        else:
            lstm_p = 0.0

        # Backpropagation
        if backprop_model is not None:
            backprop_p = float(backprop_model.predict(scaled_data)[0][0])
            results["Backpropagation"] = backprop_p
        else:
            backprop_p = 0.0
            
        # K-Means
        if kmeans_model is not None:
            km_p = int(kmeans_model.predict(scaled_data[:, :4])[0])
        else:
            km_p = 0

        # Validasi jika sama sekali tidak ada model AI yang aktif
        if not results:
            return render_template(
                'predict.html', 
                weather_status="⚠️ Gagal: Seluruh berkas model (.h5 atau .pkl) gagal dimuat oleh sistem server.",
                lr_prediction='error', ann_prediction=0, lstm_prediction=0, backprop_prediction=0, kmeans_prediction=0,
                best_model="Model Check Failure"
            )

        # 5. Logika Pemilihan "Best Model" Secara Dinamis dari model yang aktif
        avg_all = sum(results.values()) / len(results)
        dynamic_best_model = min(results, key=lambda k: abs(results[k] - avg_all))

        # 6. Penentuan Status Cuaca
        if avg_all < 20:
            status = "☀️ Cerah"
        elif avg_all < 50:
            status = "☁️ Mendung"
        elif avg_all < 80:
            status = "🌧️ Hujan"
        else:
            status = "⛈️ Badai"

        # Variabel lr_prediction dikondisikan agar tidak bernilai None murni agar lolos sensor if HTML kamu
        return render_template(
            'predict.html',
            weather_status=status,
            lr_prediction=round(lr_p, 2
