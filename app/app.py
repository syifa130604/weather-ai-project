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

@app.route('/predict-page')
def predict_page():
    return render_template('predict.html')

@app.route('/visualization')
def visualization():
    return render_template('visualization.html')

# =========================================
# ROUTE - PREDICTION LOGIC (DINAMIS)
# =========================================

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Validasi jika model keras atau scaler gagal dimuat
        if None in [lr_model, ann_model, lstm_model, backprop_model, kmeans_model, scaler]:
            return render_template('predict.html', weather_status="Error: Model AI belum siap atau gagal dimuat.")

        # 1. Mengambil input dari form
        inputs = [
            float(request.form['temp_avg']),
            float(request.form['temp_max']),
            float(request.form['temp_min']),
            float(request.form['press']),
            float(request.form['humid_avg']),
            float(request.form['ws_avg']),
            float(request.form['max_ws']),
            float(request.form['light_hour'])
        ]
        
        # 2. Preprocessing & Scaling
        data_array = np.array([inputs])
        scaled_data = scaler.transform(data_array)

        # 3. Eksekusi Prediksi Berbagai Algoritma
        lr_p = float(lr_model.predict(scaled_data)[0])
        ann_p = float(ann_model.predict(scaled_data)[0][0])
        lstm_in = scaled_data.reshape((scaled_data.shape[0], 1, scaled_data.shape[1]))
        lstm_p = float(lstm_model.predict(lstm_in)[0][0])
        bp_p = float(backprop_model.predict(scaled_data)[0][0])
        
        # K-Means (Clustering menggunakan 4 fitur utama)
        km_p = int(kmeans_model.predict(scaled_data[:, :4])[0])

        # 4. Logika Pemilihan "Best Model" Secara Dinamis
        # Kita bandingkan hasil prediksi dan cari yang paling stabil (mendekati rata-rata konsensus)
        results = {
            "Regresi Linear": lr_p,
            "ANN (Neural Network)": ann_p,
            "LSTM (RNN)": lstm_p,
            "Backpropagation": bp_p
        }
        
        avg_all = sum(results.values()) / len(results)
        # Mencari model yang selisihnya paling kecil dengan rata-rata keseluruhan
        dynamic_best_model = min(results, key=lambda k: abs(results[k] - avg_all))

        # 5. Penentuan Status Cuaca (Untuk Animasi Visual)
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
            lr_prediction=round(lr_p, 2),
            ann_prediction=round(ann_p, 2),
            lstm_prediction=round(lstm_p, 2),
            backprop_prediction=round(bp_p, 2),
            kmeans_prediction=km_p,
            best_model=dynamic_best_model # Hasil ini akan berubah sesuai data input
        )
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Prediction Error:\n{error_details}")
        # Mengembalikan string error spesifik ke template agar langsung terbaca di halaman web
        return render_template('predict.html', weather_status=f"Error: {str(e)}")

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
    # Menambahkan host='0.0.0.0' dan port agar kompatibel jika di-run di Hugging Face local space docker / server backend
    app.run(host='0.0.0.0', port=5000, debug=True)
