from flask import Flask, render_template, request, url_for
import random
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import re
from collections import defaultdict, Counter

app = Flask(__name__)

# ==============================
# 🧑‍🎓 Data Identitas Mahasiswa
# ==============================
NAMA = "Andika Rizky Darmansyah"
NPM = "50422217"
KELAS = "4IA03"

# ==============================
# 💬 Dataset sederhana untuk prediksi kata
# ==============================
CORPUS = """
Deep learning adalah cabang dari machine learning yang menggunakan neural networks
dengan banyak lapisan untuk mempelajari representasi data yang kompleks.
Deep learning digunakan untuk berbagai aplikasi seperti pengenalan gambar,
pemrosesan bahasa alami, dan prediksi data masa depan.
"""

def build_trigrams(corpus):
    tokens = re.findall(r"\w+|[^\w\s]", corpus.lower())
    bigrams = defaultdict(Counter)
    trigrams = defaultdict(Counter)
    rev_trigrams = defaultdict(Counter)

    for i in range(len(tokens) - 2):
        w1, w2, w3 = tokens[i], tokens[i+1], tokens[i+2]
        bigrams[w1][w2] += 1
        trigrams[(w1, w2)][w3] += 1
        rev_trigrams[(w2, w3)][w1] += 1

    return trigrams, bigrams, rev_trigrams

TRIGRAMS, BIGRAMS, REV_TRIGRAMS = build_trigrams(CORPUS)

# ==============================
# 🏠 Halaman Utama
# ==============================
@app.route('/')
def index():
    tugas = [
        {"title": "🧮 Kalkulator Logika", "url": url_for('kalkulator')},
        {"title": "💬 Prediksi Kata (Bidirectional)", "url": url_for('prediksi_kata')},
        {"title": "📈 Prediksi Harga Saham (Real-Time)", "url": url_for('prediksi_saham')}
    ]
    return render_template(
        'index.html',
        title="Daftar Tugas",
        tugas=tugas,
        nama=NAMA,
        npm=NPM,
        kelas=KELAS
    )

# ==============================
# 🧮 Kalkulator Logika
# ==============================
@app.route('/kalkulator', methods=['GET', 'POST'])
def kalkulator():
    hasil = None
    operasi = None
    tabel_kebenaran = []

    if request.method == 'POST':
        try:
            a = int(request.form.get('a', 0))
            b = int(request.form.get('b', 0))
            operasi = request.form.get('operasi')

            if a not in [0, 1] or b not in [0, 1]:
                hasil = "⚠️ Nilai A dan B harus berupa 0 atau 1."
            else:
                if operasi == "AND":
                    hasil = a & b
                    tabel_kebenaran = [
                        (0, 0, 0),
                        (0, 1, 0),
                        (1, 0, 0),
                        (1, 1, 1)
                    ]
                elif operasi == "OR":
                    hasil = a | b
                    tabel_kebenaran = [
                        (0, 0, 0),
                        (0, 1, 1),
                        (1, 0, 1),
                        (1, 1, 1)
                    ]
                elif operasi == "XOR":
                    hasil = a ^ b
                    tabel_kebenaran = [
                        (0, 0, 0),
                        (0, 1, 1),
                        (1, 0, 1),
                        (1, 1, 0)
                    ]
                elif operasi == "NOT":
                    hasil = 1 - a
                    tabel_kebenaran = [
                        (0, 1),
                        (1, 0)
                    ]
                else:
                    hasil = "❌ Operasi tidak dikenali."

        except Exception as e:
            hasil = f"Terjadi kesalahan: {str(e)}"

    return render_template(
        'kalkulator.html',
        title="Kalkulator Logika",
        hasil=hasil,
        operasi=operasi,
        tabel_kebenaran=tabel_kebenaran
    )


# ==============================
# 💬 Prediksi Kata
# ==============================
@app.route('/prediksi_kata', methods=['GET', 'POST'])
def prediksi_kata():
    hasil = None
    if request.method == 'POST':
        teks_input = request.form.get('teks')
        kemungkinan = ["baik", "buruk", "cerah", "gelap", "menyenangkan", "berhasil", "menarik"]
        hasil = f"{teks_input} {random.choice(kemungkinan)}"

    return render_template(
        'prediksi_kata.html',
        hasil=hasil,
        title="Prediksi Kata",
        nama=NAMA,
        npm=NPM,
        kelas=KELAS
    )

# ==============================
# 📈 Prediksi Harga Saham (Real-Time)
# ==============================
@app.route('/prediksi_saham', methods=['GET', 'POST'])
def prediksi_saham():
    # imports lokal (pastikan paket terinstall)
    import yfinance as yf
    from sklearn.linear_model import LinearRegression
    import numpy as np
    import matplotlib.pyplot as plt
    import io, base64, os

    grafik = None
    hasil_prediksi = None
    last_price = None
    symbol = None
    error = None

    if request.method == 'POST':
        symbol = (request.form.get('symbol') or '').strip().upper()
        if not symbol:
            error = "Silakan masukkan simbol saham."
        else:
            try:
                # ambil data 6 bulan terakhir
                data = yf.download(symbol, period="6mo", interval="1d")
                data = data[['Close']].dropna()

                if data.empty:
                    error = f"⚠️ Data saham untuk simbol '{symbol}' tidak ditemukan."
                else:
                    # siapkan fitur (hari ke-n)
                    X = np.arange(len(data)).reshape(-1, 1)
                    y = data['Close'].values

                    model = LinearRegression().fit(X, y)

                    # Prediksi untuk 1 hari ke depan (besok)
                    next_day = np.array([[len(data)]])
                    y_pred = model.predict(next_day)   # hasil berupa array, mis: array([123.45])
                    prediksi_besok = float(y_pred[0])  # ambil scalar

                    # harga terakhir (scalar)
                    harga_terakhir = float(y[-1])

                    # format dengan simbol dolar
                    hasil_prediksi = f"${prediksi_besok:,.2f}"
                    last_price = f"${harga_terakhir:,.2f}"

                    # buat grafik historis + titik prediksi besok
                    plt.figure(figsize=(8, 4))
                    plt.plot(data.index, data['Close'], label='Harga Historis')
                    # plot prediksi sebagai titik pada tanggal setelah data terakhir
                    try:
                        # buat x untuk plot: gunakan index + 1 hari pada tanggal
                        last_date = data.index[-1]
                        if hasattr(last_date, 'to_pydatetime'):
                            next_date = last_date + np.timedelta64(1, 'D')
                        else:
                            # fallback numeric x if index bukan datetime
                            next_date = len(data)
                    except Exception:
                        next_date = len(data)

                    # Jika index adalah datetime, plot using dates; else use numeric x
                    if np.issubdtype(type(data.index[0]), np.datetime64) or hasattr(data.index[0], 'to_pydatetime'):
                        plt.scatter([next_date], [prediksi_besok], color='red', label='Prediksi Besok')
                    else:
                        plt.scatter([len(data)], [prediksi_besok], color='red', label='Prediksi Besok')

                    plt.title(f"Harga Saham {symbol}")
                    plt.xlabel("Tanggal")
                    plt.ylabel("Harga ($)")
                    plt.legend()
                    plt.tight_layout()

                    # simpan ke base64
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    grafik = base64.b64encode(buf.getvalue()).decode('utf-8')
                    plt.close()

            except Exception as e:
                error = f"Terjadi kesalahan: {str(e)}"

    return render_template(
        'prediksi_saham.html',
        title="Prediksi Saham",
        grafik=grafik,
        hasil_prediksi=hasil_prediksi,
        last_price=last_price,
        symbol=symbol,
        error=error
    )


# ==============================
# 🚀 Jalankan Aplikasi
# ==============================
if __name__ == '__main__':
    app.run(debug=True)
