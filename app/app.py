from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import os

# Mengamankan import tensorflow dari issue memory limit / version mismatch di Hugging Face
try:
    from tensorflow.keras.models import load_model
except Exception as e:
    load_model = None

# =========================================
# INITIALIZATION
# =========================================
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models')
DOCS_PATH = os.path.join(BASE_DIR, '..', 'docs')

# Inisialisasi variabel global untuk menampung arsitektur model
lr_model = ann_model = lstm_model = backprop_model = kmeans_model = scaler = None

# =========================================
# LOAD MODELS & SCALER (SAFE LOADING)
# =========================================
try:
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
    print(f"Safe Load Warning: {str(e)}")

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
# ROUTE - PREDICTION SYSTEM (ALL-IN-ONE)
# =========================================

@app.route('/predict-page', methods=['GET', 'POST'])
def predict_page():
    # 1. JIKA DIAKSES BIASA (GET METHOD)
    if request.method == 'GET':
        return render_template('predict.html', weather_status=None, lr_prediction=None)

    # 2. JIKA DIKLIK TOMBOL ANALISIS (POST METHOD)
    try:
        # Cek ketersediaan Scaler
        if scaler is None:
            return render_template(
                'predict.html',
                weather_status="⚠️ Gagal: Berkas 'scaler.pkl' tidak terbaca di server backend.",
                lr_prediction='error', ann_prediction=0, lstm_prediction=0, backprop_prediction=0, kmeans_prediction=0, best_model="System"
            )

        # Mengambil data input formulir secara aman dengan default value 0.0 jika kosong
        temp_avg = float(request.form.get('temp_avg', 0) or 0)
        temp_max = float(request.form.get('temp_max', 0) or 0)
        temp_min = float(request.form.get('temp_min', 0) or 0)
        press = float(request.form.get('press', 0) or 0)
        humid_avg = float(request.form.get('humid_avg', 0) or 0)
        ws_avg = float(request.form.get('ws_avg', 0) or 0)
        max_ws = float(request.form.get('max_ws', 0) or 0)
        light_hour = float(request.form.get('light_hour', 0) or 0)

        inputs = [temp_avg, temp_max, temp_min, press, humid_avg, ws_avg, max_ws, light_hour]
        
        # Transformasi Data
        data_array = np.array([inputs])
        scaled_data = scaler.transform(data_array)

        results = {}

        # Kalkulasi Linear Regression (Sebagai backup utama jika Deep Learning crash)
        if lr_model is not None:
            lr_p = float(lr_model.predict(scaled_data)[0])
            results["Regresi Linear"] = lr_p
        else:
            lr_p = 0.0

        # Kalkulasi ANN
        if ann_model is not None:
            ann_p = float(ann_model.predict(scaled_data)[0][0])
            results["ANN (Neural Network)"] = ann_p
        else:
            ann_p = 0.0

        # Kalkulasi LSTM
        if lstm_model is not None:
            lstm_in = scaled_data.reshape((scaled_data.shape[0], 1, scaled_data.shape[1]))
            lstm_p = float(lstm_model.predict(lstm_in)[0][0])
            results["LSTM (RNN)"] = lstm_p
        else:
            lstm_p = 0.0

        # Kalkulasi Backpropagation
        if backprop_model is not None:
            bp_p = float(backprop_model.predict(scaled_data)[0][0])
            results["Backpropagation"] = bp_p
        else:
            bp_p = 0.0

        # Kalkulasi K-Means
        km_p = int(kmeans_model.predict(scaled_data[:, :4])[0]) if kmeans_model is not None else 0

        # Proteksi jika seluruh model gagal dimuat
        if not results:
            return render_template(
                'predict.html',
                weather_status="⚠️ Gagal: Seluruh model (.h5/.pkl) gagal dimuat di server Hugging Face.",
                lr_prediction='error', ann_prediction=0, lstm_prediction=0, backprop_prediction=0, kmeans_prediction=0, best_model="System"
            )

        # Menghitung Rata-rata Konsensus Penentuan Status Cuaca
        avg_all = sum(results.values()) / len(results)
        dynamic_best_model = min(results, key=lambda k: abs(results[k] - avg_all))

        if avg_all < 20:
            status = "☀️ Cerah"
        elif avg_all < 50:
            status = "☁️ Mendung"
        elif avg_all < 80:
            status = "🌧️ Hujan"
        else:
            status = "⛈️ Badai"

        return render_template(
            'predict.html',
            weather_status=status,
            lr_prediction=round(lr_p, 2) if lr_p != 0 else 0.01,
            ann_prediction=round(ann_p, 2),
            lstm_prediction=round(lstm_p, 2),
            backprop_prediction=round(bp_p, 2),
            kmeans_prediction=km_p,
            best_model=dynamic_best_model
        )

    except Exception as e:
        import traceback
        print(f"Fatal Prediction Crash:\n{traceback.format_exc()}")
        return render_template(
            'predict.html',
            weather_status=f"⚠️ Gagal Eksekusi: {str(e)}",
            lr_prediction='error', ann_prediction=0, lstm_prediction=0, backprop_prediction=0, kmeans_prediction=0, best_model="Crash Handler"
        )

# =========================================
# ROUTE - MODEL COMPARISON
# =========================================

@app.route('/comparison')
def comparison():
    csv_file = os.path.join(DOCS_PATH, 'model_comparison.csv')
    try:
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            comparison_data = df.to_dict(orient='records')
            return render_template('comparison.html', comparison=comparison_data)
        else:
            return render_template('comparison.html', comparison=[])
    except Exception as e:
        print(f"Error Reading CSV: {e}")
        return render_template('comparison.html', comparison=[])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=True)
