# Proje Adı: Su Altı Nesne Tespiti ve Gemini ile Tanımlama

Bu proje, Yapay Zeka ve Teknoloji Akademisi Hackathon'u için geliştirilmiştir. Uygulama, kullanıcı tarafından yüklenen su altı görüntülerindeki nesneleri YOLOv8 modeli ile tespit eder. Ardından, tespit edilen ana nesne (veya tüm resim) Google Gemini API kullanılarak tanımlanır ve nesne/canlı hakkında ekolojik bilgiler, istilacılık durumu ve eğitici notlar sunulur.

## Kullanılan Teknolojiler

* **Backend:** Python, FastAPI, Uvicorn
* **Nesne Tespiti:** Ultralytics YOLOv8
* **Yapay Zeka Analizi:** Google Gemini API (`gemini-1.5-flash-latest`)
* **Görüntü İşleme:** OpenCV, Pillow
* **Frontend:** HTML, CSS
* **Şablon Motoru:** Jinja2
* **API Anahtar Yönetimi:** python-dotenv

## Kurulum Adımları

1.  **Repository'yi Klonlayın veya İndirin:**
    Bu repository'yi bilgisayarınıza klonlayın:
    ```bash
    git clone [https://github.com/SENIN_KULLANICI_ADIN/SENIN_REPO_ADIN.git](https://github.com/SENIN_KULLANICI_ADIN/SENIN_REPO_ADIN.git)
    cd SENIN_REPO_ADIN
    ```
    (VEYA ZIP olarak indirip bir klasöre çıkartın.)

2.  **Python Sanal Ortamı Oluşturun ve Aktif Edin:**
    Proje ana dizinindeyken:
    ```bash
    python -m venv venv
    ```
    Aktif etmek için:
    * Windows (Command Prompt): `.\venv\Scripts\activate.bat`
    * macOS/Linux: `source venv/bin/activate`

3.  **Gerekli Python Kütüphanelerini Yükleyin:**
    Sanal ortam aktifken:
    ```bash
    pip install -r requirements.txt
    ```

4.  **API Anahtarını Ayarlayın:**
    * Proje ana dizininde `.env` adında bir dosya oluşturun.
    * Google AI Studio'dan aldığınız Gemini API anahtarınızı bu dosyaya aşağıdaki formatta ekleyin:
        ```env
        GEMINI_API_KEY=BURAYA_KENDI_API_ANAHTARINIZI_YAZIN
        ```

## Uygulamayı Çalıştırma

1.  Sanal ortamınızın aktif olduğundan emin olun.
2.  Proje ana dizinindeyken terminalde aşağıdaki komutu çalıştırın:
    ```bash
    uvicorn main:app --reload
    ```
3.  İnternet tarayıcınızı açın ve `http://127.0.0.1:8000` adresine gidin.

## Projenin Özellikleri

* Kullanıcı tarafından resim veya video dosyası yüklenebilir.
* Yüklenen resimlerde YOLOv8 ile nesne tespiti yapılır ve en belirgin nesnenin etrafına "Analiz Ediliyor..." etiketiyle bir kutu çizilir.
* Tespit edilen ana nesne (veya YOLO bir şey bulamazsa tüm resim) Google Gemini API'sine gönderilerek tanımlanır.
* Gemini'den gelen açıklama (canlının türü, ekolojik rolü, istilacı olup olmadığı, eğitici bilgiler vb.) kullanıcı arayüzünde gösterilir.
* Videolar için temel nesne tespiti ve etiketsiz kutu çizimi yapılır (Gemini entegrasyonu şimdilik sadece resimler için aktiftir).

## Hackathon Temaları ile İlişkisi

* **Sürdürülebilirlik Çözümleri:** Uygulama, deniz canlılarının (özellikle potansiyel istilacı türlerin) Gemini API aracılığıyla ekolojik etkilerini analiz ederek deniz ekosistemleri hakkında farkındalık yaratmayı hedefler.
* **Yeni Nesil Öğrenme:** Kullanıcıların yükledikleri su altı görsellerindeki canlıları ve nesneleri, YOLO ile ön tespit ve Gemini ile detaylı tanımlama ve eğitici notlar sayesinde interaktif bir şekilde öğrenmelerini sağlar.

---
*Lütfen yukarıdaki `SENIN_KULLANICI_ADIN` ve `SENIN_REPO_ADIN` kısımlarını kendi GitHub kullanıcı adınız ve repository adınız ile değiştirmeyi unutmayın.*
