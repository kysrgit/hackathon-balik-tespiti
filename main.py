import google.generativeai as genai
from PIL import Image # Görüntü işleme için Pillow kütüphanesi
import io # Görüntüyü byte formatına çevirmek için
from dotenv import load_dotenv # .env dosyasını yüklemek için
import os
import shutil
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from typing import Optional, List # List import edilmiş olmalı
import uuid
import cv2 # Hala video işleme için gerekli olabilir
from pathlib import Path # Path nesnesi için import

# Ultralytics kütüphanesini import et
from ultralytics import YOLO

# --- Sabitler ve Klasörler (Path nesneleriyle) ---
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
DETECTION_DIR = BASE_DIR / "static" / "detections"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DETECTION_DIR.mkdir(parents=True, exist_ok=True)
# -------------------------------------------------

# .env dosyasındaki değişkenleri yükle (özellikle GEMINI_API_KEY)
load_dotenv() 
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Gemini API'sini yapılandır
gemini_configured = False # Gemini yapılandırma bayrağı
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_configured = True # Başarılı olursa True yap
        print("Gemini API anahtarı başarıyla yüklendi ve yapılandırıldı.")
    except Exception as e_gemini_config:
        print(f"HATA: Gemini API yapılandırılırken sorun oluştu: {e_gemini_config}")
else:
    print("UYARI: GEMINI_API_KEY .env dosyasında bulunamadı veya yüklenemedi.")

app = FastAPI()

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static_files")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# YOLO Modelini Yükleme
model = None
MODEL_CLASSES = {}
try:
    model = YOLO("yolov8n.pt") 
    print("YOLOv8n modeli başarıyla yüklendi.")
    if model and hasattr(model, 'names'):
        MODEL_CLASSES = model.names
    else:
        print("UYARI: Model sınıf isimleri yüklenemedi veya model None.")
except Exception as e:
    print(f"HATA: YOLO modeli yüklenirken sorun oluştu: {e}")

def get_file_type(content_type: Optional[str], filename: Optional[str]) -> str:
    """Dosya içerik türüne veya dosya adına göre 'image' veya 'video' döndürür."""
    if content_type:
        if content_type.startswith("image"):
            return "image"
        elif content_type.startswith("video"):
            return "video"
    if filename:
        ext = os.path.splitext(filename)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
            return "image"
        elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            return "video"
    return "unknown"

async def run_yolo_detection_simplified(file_path_on_server_str: str, original_filename: str, content_type_from_upload: Optional[str]): # file_path_on_server string olarak alınacak
    """
    YOLO'yu sadece yer bulucu olarak kullanır, en iyi tespiti Gemini'ye sorar.
    YOLO'nun yanlış etiketini resme YAZMAZ.
    """
    global model, MODEL_CLASSES, gemini_configured # gemini_configured'ı global olarak ekle

    if model is None:
        return None, ["YOLO modeli yüklenemedi."], None, "YOLO modeli yüklenemedi."

    yolo_raw_summary = [] 
    processed_output_path_str = None 
    gemini_description = None 
    error_message = None
    user_facing_summary = [] 

    file_path_on_server = Path(file_path_on_server_str) # String'i Path nesnesine çevir

    file_type = get_file_type(content_type_from_upload, original_filename)
    print(f"İşlenecek dosya tipi: {file_type}")

    file_extension = file_path_on_server.suffix # Path nesnesinden al
    detected_filename_base = f"detected_{uuid.uuid4()}"
    processed_output_filename = f"{detected_filename_base}{file_extension}" if file_type == "image" else f"{detected_filename_base}.mp4"
    processed_output_path_obj = DETECTION_DIR / processed_output_filename # Path nesnesiyle oluştur
    
    CONFIDENCE_THRESHOLD_YOLO_BOX_DRAW = 0.25 
    
    best_yolo_box_for_gemini = None 
    highest_yolo_confidence = 0.0     
    original_image_for_processing = None
    image_to_send_to_gemini = None # <<<<---- DEĞİŞKENİ BURADA NONE OLARAK BAŞLAT

    try:
        if file_type == "image":
            original_image_for_processing = cv2.imread(str(file_path_on_server)) # cv2 string bekler
            if original_image_for_processing is None:
                raise ValueError(f"Görüntü okunamadı: {file_path_on_server}")

            annotated_frame = original_image_for_processing.copy()
            results = model(original_image_for_processing) 

            if results and results[0].boxes:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                
                for i in range(len(boxes)):
                    confidence = confidences[i]
                    if confidence >= CONFIDENCE_THRESHOLD_YOLO_BOX_DRAW: 
                        if confidence > highest_yolo_confidence:
                            highest_yolo_confidence = confidence
                            best_yolo_box_for_gemini = boxes[i]
                        
                        yolo_class_id = int(results[0].boxes.cls[i].cpu())
                        yolo_class_name = MODEL_CLASSES.get(yolo_class_id, f"ID_{yolo_class_id}")
                        yolo_raw_summary.append(f"YOLO DEBUG: {yolo_class_name} ({confidence:.2f})")

                if best_yolo_box_for_gemini is not None:
                    x1_draw, y1_draw, x2_draw, y2_draw = map(int, best_yolo_box_for_gemini)
                    cv2.rectangle(annotated_frame, (x1_draw, y1_draw), (x2_draw, y2_draw), (0, 255, 0), 3) 
                    cv2.putText(annotated_frame, "Analiz Ediliyor...", (x1_draw, y1_draw - 10 if y1_draw -10 > 10 else y1_draw + 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                else: 
                    yolo_raw_summary.append("YOLO çizilecek kadar güvenli bir kutu bulamadı.")

                cv2.imwrite(str(processed_output_path_obj), annotated_frame)
                processed_output_path_str = str(processed_output_path_obj)
                print(f"Resim işlendi (etiketsiz kutu) ve kaydedildi: {processed_output_path_str}")
                print(f"YOLO Ham Özet (Debug): {yolo_raw_summary}")
            
            else: 
                yolo_raw_summary.append("YOLO resimde kutu bulamadı.")
                print("YOLO resimde kutu bulamadı. Orijinal resim Gemini'ye gönderilecek.")
                shutil.copy(str(file_path_on_server), str(processed_output_path_obj))
                processed_output_path_str = str(processed_output_path_obj)

            # --- GEMINI VISION ÇAĞRISI ---
            if best_yolo_box_for_gemini is not None and original_image_for_processing is not None:
                print(f"En iyi YOLO kutusu ({highest_yolo_confidence:.2f}) Gemini için kırpılıyor...")
                x1_c, y1_c, x2_c, y2_c = map(int, best_yolo_box_for_gemini)
                h_img, w_img = original_image_for_processing.shape[:2]
                x1_c, y1_c = max(0, x1_c), max(0, y1_c)
                x2_c, y2_c = min(w_img, x2_c), min(h_img, y2_c)
                
                if x1_c < x2_c and y1_c < y2_c: 
                    cropped_img = original_image_for_processing[y1_c:y2_c, x1_c:x2_c]
                    image_to_send_to_gemini = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
                else: 
                    print("UYARI: Gemini için kırpma kutusu geçersiz, tüm resim gönderilecek.")
                    if original_image_for_processing is not None: # Bu kontrol zaten yukarıda yapıldı ama tekrar zarar vermez
                        image_to_send_to_gemini = Image.fromarray(cv2.cvtColor(original_image_for_processing, cv2.COLOR_BGR2RGB))
            elif original_image_for_processing is not None: 
                print("YOLO kutu bulamadı, tüm resim Gemini'ye gönderiliyor.")
                image_to_send_to_gemini = Image.fromarray(cv2.cvtColor(original_image_for_processing, cv2.COLOR_BGR2RGB))

            if image_to_send_to_gemini and GEMINI_API_KEY and gemini_configured: 
                try:
                    vision_model = genai.GenerativeModel('gemini-1.5-flash-latest')
                    prompt_text = (
                        "Bu görüntüdeki ana deniz canlısını veya nesneyi tanımlayın. "
                        "Eğer bir canlıysa, yaygın adını ve bilimsel türünü belirtin. "
                        "Bu tür, özellikle Türkiye denizleri veya Akdeniz ekosistemleri için istilacı bir tür müdür? "
                        "Kısaca ekolojik rolünü veya sürdürülebilirlik üzerindeki etkisini açıklayın. "
                        "Son olarak, bu canlı/nesne hakkında ilginç bir eğitici bilgi ekleyin."
                    )
                    response = vision_model.generate_content([prompt_text, image_to_send_to_gemini], request_options={"timeout": 90})
                    
                    if response and hasattr(response, 'text') and response.text:
                        gemini_description = response.text
                        user_facing_summary.append("Gemini Analizi: Başarılı.") 
                        print("Gemini (tematik) yanıtı alındı.")
                    elif response and not response.text and response.candidates and response.candidates[0].finish_reason.name != "SAFETY":
                        gemini_description = "Gemini bir metin üretmedi (belki resim boş veya anlamsız?)."
                        user_facing_summary.append("Gemini Analizi: Boş yanıt.")
                    elif response and response.candidates and response.candidates[0].finish_reason.name == "SAFETY":
                        gemini_description = "Gemini güvenlik filtreleri nedeniyle yanıt üretemedi."
                        user_facing_summary.append("Gemini Analizi: Güvenlik filtresi.")
                    else:
                        gemini_description = "Gemini'den geçerli bir yanıt veya metin alınamadı."
                        user_facing_summary.append("Gemini Analizi: Yanıt yok.")
                except Exception as e_gemini:
                    gemini_description = f"Gemini ile iletişimde hata: {str(e_gemini)}"
                    user_facing_summary.append("Gemini Analizi: Hata.")
                    print(f"HATA - Gemini API çağrısı: {e_gemini}")
            elif not (GEMINI_API_KEY and gemini_configured):
                 gemini_description = "Gemini API anahtarı yapılandırılmadığı için analiz yapılamadı."
                 user_facing_summary.append("Gemini Analizi: API anahtarı yok.")
            elif not image_to_send_to_gemini: # Bu durum artık daha az olası
                 gemini_description = "Gemini'ye gönderilecek görüntü hazırlanamadı."
                 user_facing_summary.append("Gemini Analizi: Görüntü yok.")
        
        elif file_type == "video":
            cap = cv2.VideoCapture(str(file_path_on_server)) # cv2 string bekler
            if not cap.isOpened(): raise ValueError(f"Video açılamadı: {file_path_on_server}")
            
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0: fps = 20 

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_video = cv2.VideoWriter(str(processed_output_path_obj), fourcc, fps, (frame_width, frame_height)) # cv2 string bekler
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret: break
                frame_count += 1
                annotated_frame_video = frame.copy() 
                
                try: 
                    video_results = model(frame) 
                    if video_results and video_results[0].boxes:
                        boxes_vid = video_results[0].boxes.xyxy.cpu().numpy()
                        confidences_vid = video_results[0].boxes.conf.cpu().numpy()
                        for i_vid in range(len(boxes_vid)):
                            if confidences_vid[i_vid] >= CONFIDENCE_THRESHOLD_YOLO_BOX_DRAW:
                                x1v, y1v, x2v, y2v = map(int, boxes_vid[i_vid])
                                cv2.rectangle(annotated_frame_video, (x1v, y1v), (x2v, y2v), (0, 255, 0), 2)
                        
                        if frame_count <= 3: 
                            for box_obj in video_results[0].boxes:
                                confidence_s = float(box_obj.conf[0])
                                if confidence_s >= CONFIDENCE_THRESHOLD_YOLO_BOX_DRAW:
                                    class_id_s = int(box_obj.cls[0])
                                    class_name_s = MODEL_CLASSES.get(class_id_s, f"ID_{class_id_s}")
                                    yolo_raw_summary.append(f"K{frame_count} YOLO: {class_name_s} ({confidence_s:.2f})") 
                except Exception as e_frame_vid:
                     print(f"HATA: Video karesi işlenirken (Kare {frame_count}): {e_frame_vid}")
                
                out_video.write(annotated_frame_video) 
            
            cap.release()
            out_video.release()
            processed_output_path_str = str(processed_output_path_obj)
            if frame_count == 0: user_facing_summary.append("Video dosyası boş veya okunamadı.")
            else: user_facing_summary.append(f"Video işlendi ({frame_count} kare). Yalnızca resimler için Gemini analizi yapılır.") 
            gemini_description = "Video analizi için Gemini bu versiyonda kullanılmamaktadır." 
            print(f"Video işlendi (etiketsiz kutular) ve kaydedildi: {processed_output_path_str}")
        
        else: 
            error_message = f"Desteklenmeyen dosya türü: {file_type}"
            user_facing_summary.append(error_message)
            print(error_message)

    except Exception as e:
        print(f"HATA - run_yolo_detection_simplified genel: {e}")
        error_message = f"İşlem sırasında genel hata: {str(e)}"
        user_facing_summary.append(f"Genel Hata: {str(e)}")
        if processed_output_path_obj and processed_output_path_obj.exists():
            try: 
                processed_output_path_obj.unlink() 
                print(f"Hata nedeniyle yarım kalan dosya silindi: {processed_output_path_obj}")
            except Exception as e_remove: 
                print(f"Yarım kalan dosya silinirken hata: {e_remove}")
        processed_output_path_str = None

    if not user_facing_summary and not error_message:
        user_facing_summary.append("İşlem tamamlandı, ancak görüntülenecek özel bir sonuç bulunamadı.")

    return processed_output_path_str, user_facing_summary, gemini_description, error_message

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    message = None
    message_type = "info"
    uploaded_file_name_for_html = None
    uploaded_file_type_display = None
    detection_result_filename_for_html = None
    detected_file_type_display = None
    gemini_info_display = None
    detection_summary = [] # Bu, user_facing_summary olacak
    original_unique_filename = "unknown_original" 
    file_path_on_server_str = None # String olarak saklayalım

    try:
        if not file.filename or not file.filename.strip():
            return templates.TemplateResponse("index.html", {
                "request": request,
                "message": "Geçersiz veya boş dosya adı.",
                "message_type": "error"
            })

        print(f"Yüklenen dosya: {file.filename}, Tip: {file.content_type}")

        file_extension = Path(file.filename).suffix.lower()
        file_type_from_content = get_file_type(file.content_type, file.filename)
        if not file_extension: 
            if file_type_from_content == "image": file_extension = ".jpg"
            elif file_type_from_content == "video": file_extension = ".mp4"
            else: file_extension = ".bin"
        
        original_unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path_on_server_obj = UPLOAD_DIR / original_unique_filename 
        file_path_on_server_str = str(file_path_on_server_obj) # String'e çevir

        try:
            with open(file_path_on_server_obj, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            print(f"Dosya kaydedildi: {file_path_on_server_obj}")
        except Exception as e_save:
            print(f"HATA - Dosya kaydedilirken: {e_save}")
            return templates.TemplateResponse("index.html", {
                "request": request,
                "message": f"Dosya diske kaydedilirken sorun: {str(e_save)}",
                "message_type": "error"
            })

        message = f"'{file.filename}' yüklendi. İşleniyor..."
        uploaded_file_name_for_html = f"uploads/{original_unique_filename}"
        uploaded_file_type_display = file_type_from_content
        detected_file_type_display = file_type_from_content

        if model:
            print(f"YOLO ve Gemini çağrılıyor: {file_path_on_server_str}")
            
            processed_output_path_server_str, summary, gemini_text, yolo_error = await run_yolo_detection_simplified(
                file_path_on_server_str, # String olarak gönder
                original_unique_filename, 
                file.content_type
            )
            
            detection_summary = summary if summary else [] # Bu artık user_facing_summary
            gemini_info_display = gemini_text 

            if yolo_error:
                message += f" | Hata: {yolo_error}"
                message_type = "error"
            elif processed_output_path_server_str and Path(processed_output_path_server_str).exists():
                relative_processed_path = Path(processed_output_path_server_str).relative_to(BASE_DIR / "static")
                detection_result_filename_for_html = str(relative_processed_path).replace("\\", "/")
                detected_file_type_display = get_file_type(None, os.path.basename(processed_output_path_server_str))
                message += " | İşlem tamamlandı."
                message_type = "success"
                if gemini_info_display: message += " Gemini analizi yapıldı."
                print(f"HTML'e gönderilecek tespit: {detection_result_filename_for_html}, Tipi: {detected_file_type_display}")
            else: 
                message += " | Tespit sonucu üretilemedi veya kaydedilemedi."
                if not yolo_error: message_type = "warning"
        else:
            message = "YOLO modeli yüklenemediği için tespit yapılamadı."
            message_type = "error"

        template_context = {
            "request": request, "message": message, "message_type": message_type,
            "uploaded_file_name": uploaded_file_name_for_html,
            "uploaded_file_type": uploaded_file_type_display,
            "detection_result_name": detection_result_filename_for_html, 
            "detected_file_type": detected_file_type_display,
            "gemini_info": gemini_info_display, 
            "detection_summary": detection_summary # Bu artık user_facing_summary
        }
        return templates.TemplateResponse("index.html", template_context)

    except ValueError as ve:
        print(f"HATA - /upload (ValueError): {ve}")
        return templates.TemplateResponse("index.html", {"request": request, "message": f"Giriş hatası: {str(ve)}", "message_type": "error"})
    except IOError as ioe:
        print(f"HATA - /upload (IOError): {ioe}")
        return templates.TemplateResponse("index.html", {"request": request, "message": f"Dosya hatası: {str(ioe)}", "message_type": "error"})
    except Exception as e:
        print(f"HATA - /upload (Beklenmedik): {e}")
        return templates.TemplateResponse("index.html", {"request": request, "message": "Beklenmedik bir sunucu hatası oluştu.", "message_type": "error"})

# Sunucuyu çalıştırmak için terminalden `uvicorn main:app --reload` komutunu kullanın.
