from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import base64

app = Flask(__name__)

# --- AYARLAR ---
MODEL_YOLU = "geomind_local_model.keras" # Modelin tam adı

# Sınıf isimleri (Eğitimdeki alfabetik sırayla)
SINIFLAR = ['Andesite', 'Basalt', 'Coal', 'Gneiss', 'Granite', 
            'Limestone', 'Marble', 'Quartzite', 'Rhyolite', 
            'Sandstone', 'Schist']

print("🧠 Model yükleniyor... (RTX 4050 Devrede)")
try:
    # Modeli yükle
    model = tf.keras.models.load_model(MODEL_YOLU)
    print("✅ Model başarıyla yüklendi! Sunucu hazır.")
except Exception as e:
    print(f"❌ KRİTİK HATA: Model yüklenemedi! Dosya adını kontrol et: {e}")
    model = None

def resim_hazirla(img_bytes):
    # Gelen bayt verisini resme çevir
    img = Image.open(io.BytesIO(img_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    # Modelin istediği boyut (224x224)
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0 # Normalize et
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'success': False, 'detail': 'Model sunucuda yüklü değil.'}), 500
        
    try:
        # Telefondan veriyi al
        data = request.get_json(force=True)
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'success': False, 'detail': 'Resim verisi bulunamadı.'}), 400

        # Base64 çözme ve resmi hazırlama
        img_bytes = base64.b64decode(image_data)
        processed_image = resim_hazirla(img_bytes)
        
        # Tahmin yap
        predictions = model.predict(processed_image)
        en_yuksek_index = np.argmax(predictions)
        guven_orani = float(predictions[0][en_yuksek_index] * 100)
        
        # --- 🛡️ GÜVENLİK DUVARI (THRESHOLD) ---
        # Eğer güven oranı %60'ın altındaysa, tahmin yapma!
        if guven_orani < 60.0:
            print(f"⚠️ Düşük Güven: %{guven_orani:.1f} (Reddedildi)")
            return jsonify({
                'success': True,
                'data': {
                    'isim': 'Tanımlanamadı ❓',
                    'detay': f"Bu görüntüden tam emin olamadım (Güven: %{guven_orani:.1f}).\nLütfen taşı daha yakından, iyi bir ışıkta ve net çekmeyi dene."
                }
            })

        # Eğer %60'ın üstündeyse normal devam et
        kazanan_tas = SINIFLAR[en_yuksek_index]
        
        # Basit Bilgi Bankası
        tas_bilgileri = {
            'Andesite': 'Gri/Siyah volkanik kayaç. İnşaat ve yol yapımında kullanılır.',
            'Basalt': 'Koyu renkli, sert volkanik kaya. Parke taşı olarak yaygındır.',
            'Coal': 'Kömür. Organik tortul kayaç, enerji kaynağıdır.',
            'Gneiss': 'Şeritli yapıda metamorfik kayaç. Granitten dönüşmüştür.',
            'Granite': 'Sert, kristalli magmatik kayaç. Mutfak tezgahlarında sıkça görülür.',
            'Limestone': 'Kireç taşı. İçinde fosil bulunabilir, çimento yapımında kullanılır.',
            'Marble': 'Mermer. Kireç taşının başkalaşım geçirmiş halidir.',
            'Quartzite': 'Kuvarsit. Çok sert ve dayanıklı bir başkalaşım kayacıdır.',
            'Rhyolite': 'Açık renkli, silisli volkanik kayaç.',
            'Sandstone': 'Kum taşı. Yapılarda ve süslemelerde kullanılır.',
            'Schist': 'Şist. Yapraklı yapıda, kolay ayrılabilen metamorfik kayaç.'
        }
        
        detay_bilgi = tas_bilgileri.get(kazanan_tas, "Bu taş hakkında detaylı bilgi veritabanında yok.")

        print(f"📸 TAHMİN: {kazanan_tas} (Güven: %{guven_orani:.1f})")

        return jsonify({
            'success': True,
            'data': {
                'isim': kazanan_tas,
                'detay': f"{detay_bilgi}\n(Güven Oranı: %{guven_orani:.1f})"
            }
        })

    except Exception as e:
        print(f"⚠️ Hata: {e}")
        return jsonify({'success': False, 'detail': str(e)}), 500

if __name__ == '__main__':
    # Sunucuyu başlat
    app.run(host='0.0.0.0', port=5000)