import os
import sys
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from collections import deque
import time
import requests
import sqlite3
from datetime import datetime
import json
import threading

# --- Configurações e Parâmetros ---
MODELS_DIR = '../trained_models'
OPTICAL_FLOW_NORMAL_DIR = '../optical_flow_data/treino'
SCREENSHOTS_DIR = '../screenshots'
THRESHOLDS_FILE = os.path.join(MODELS_DIR, 'anomaly_thresholds.txt')

# Nova base de dados expandida
DB_NAME = 'anomalies_enhanced.db'

# Dimensões dos frames de Fluxo Óptico
IMG_HEIGHT, IMG_WIDTH = 64, 64
IMG_CHANNELS = 2
SEQUENCE_LENGTH = 20  # ✅ AUMENTADO de 10 para 20

# ⚡ OTIMIZAÇÕES PARA HARDWARE LIMITADO (Core i5, 16GB RAM, sem GPU)
SKIP_FRAMES = 2  # Processar 1 a cada 3 frames (aumenta FPS)
MAX_FRAME_WIDTH = 640  # Redimensionar vídeo grande para 640px
USE_THREADING = True  # Processar predições em thread separada
BATCH_PREDICTION = False  # Desabilitar batch para economizar RAM

# API do servidor
ANOMALY_LOGGING_API = 'http://127.0.0.1:5000/api/log_anomaly'

# Configurações de screenshot
CAMERA_ID = 'CAM01'
DEFAULT_LOCATION = 'Entrada Principal'

# ✅ NOVOS PARÂMETROS PARA REDUZIR FALSOS POSITIVOS
MIN_MOVEMENT_THRESHOLD = 0.8  # Magnitude mínima de movimento para processar
ANOMALY_COOLDOWN = 3.0  # Segundos entre alertas do mesmo tipo
ANOMALY_CONFIRMATION_FRAMES = 3  # Frames consecutivos necessários para confirmar
SAFETY_FACTOR = 1.3  # Multiplicador de segurança para limiares

class EnhancedAnomalyDetector:
    def __init__(self):
        self.cae_model = None
        self.convlstm_model = None
        self.cae_threshold = None
        self.convlstm_threshold = None
        self.last_anomaly_time = 0  # ✅ Para cooldown
        self.anomaly_counter = 0  # ✅ Para confirmação
        self.init_database()
        self.create_screenshot_dirs()
        
    def init_database(self):
        """Inicializa a base de dados expandida"""
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS anomalies_enhanced (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    
                    -- Dados básicos
                    timestamp TEXT NOT NULL,
                    type TEXT NOT NULL,
                    location TEXT,
                    
                    -- Scores detalhados
                    cae_score REAL,
                    convlstm_score REAL,
                    combined_score REAL,
                    
                    -- Screenshot data
                    screenshot_path TEXT,
                    screenshot_filename TEXT,
                    
                    -- Metadata técnica
                    camera_id TEXT DEFAULT 'CAM01',
                    frame_resolution TEXT,
                    risk_level TEXT,
                    
                    -- Info adicional
                    detection_duration REAL,
                    frame_number INTEGER,
                    additional_data TEXT
                );
            ''')
            conn.commit()
        print("✅ Base de dados expandida inicializada.")

    def create_screenshot_dirs(self):
        """Cria estrutura de pastas para screenshots por data"""
        if not os.path.exists(SCREENSHOTS_DIR):
            os.makedirs(SCREENSHOTS_DIR)
        
        # Criar pasta para hoje
        today = datetime.now()
        date_path = os.path.join(SCREENSHOTS_DIR, 
                                str(today.year), 
                                f"{today.month:02d}", 
                                f"{today.day:02d}")
        os.makedirs(date_path, exist_ok=True)
        print(f"✅ Diretório de screenshots criado: {date_path}")

    def load_models(self):
        """Carrega os modelos treinados"""
        print("📥 A carregar modelos treinados...")
        try:
            self.cae_model = load_model(os.path.join(MODELS_DIR, 'cae_final.keras'))
            self.convlstm_model = load_model(os.path.join(MODELS_DIR, 'convlstm_final.keras'))
            print("✅ Modelos carregados com sucesso.")
        except Exception as e:
            print(f"❌ Erro ao carregar modelos: {e}")
            sys.exit(1)

    def load_thresholds(self):
        """Carrega ou calcula limiares"""
        if os.path.exists(THRESHOLDS_FILE):
            print("📂 Carregando limiares pré-calculados...")
            try:
                with open(THRESHOLDS_FILE, 'r') as f:
                    lines = f.readlines()
                    self.cae_threshold = float(lines[0].split(':')[1])
                    self.convlstm_threshold = float(lines[1].split(':')[1])
                print(f"✅ Limiares carregados: CAE={self.cae_threshold:.4f}, ConvLSTM={self.convlstm_threshold:.4f}")
            except Exception as e:
                print(f"❌ Erro ao carregar limiares: {e}")
                self.calculate_thresholds()
        else:
            self.calculate_thresholds()

    def calculate_thresholds(self):
        """✅ MELHORADO: Calcula limiares mais conservadores"""
        print("🧮 Calculando limiares...")
        
        # Valores mais altos para reduzir falsos positivos
        self.cae_threshold = 0.095  # Aumentado de 0.05
        self.convlstm_threshold = 0.15  # Aumentado de 0.08
        
        # Salvar para próxima execução
        with open(THRESHOLDS_FILE, 'w') as f:
            f.write(f"cae_threshold:{self.cae_threshold}\n")
            f.write(f"convlstm_threshold:{self.convlstm_threshold}\n")
        
        print(f"✅ Limiares definidos: CAE={self.cae_threshold:.4f}, ConvLSTM={self.convlstm_threshold:.4f}")

    def preprocess_frame(self, gray_frame):
        """✅ NOVO: Pré-processamento para reduzir ruído"""
        # Aplicar filtro gaussiano para suavizar ruído
        smoothed = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        
        # Equalização de histograma para lidar com mudanças de iluminação
        equalized = cv2.equalizeHist(smoothed)
        
        return equalized

    def calculate_flow_magnitude(self, flow):
        """✅ NOVO: Calcula magnitude do fluxo óptico"""
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        avg_magnitude = np.mean(magnitude)
        max_magnitude = np.max(magnitude)
        return avg_magnitude, max_magnitude

    def apply_roi_mask(self, flow, margin_percent=0.1):
        """✅ NOVO: Aplica máscara para ignorar bordas"""
        h, w = flow.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        
        # Calcular margens
        margin_h = int(h * margin_percent)
        margin_w = int(w * margin_percent)
        
        # Criar máscara (1 no centro, 0 nas bordas)
        mask[margin_h:h-margin_h, margin_w:w-margin_w] = 1
        
        # Aplicar máscara
        flow[..., 0] *= mask
        flow[..., 1] *= mask
        
        return flow

    def validate_score(self, score):
        """✅ NOVO: Valida scores para evitar NaN/Inf"""
        if np.isnan(score) or np.isinf(score):
            return 0.0
        return float(score)

    def capture_screenshot_with_overlay(self, frame, anomaly_type, combined_score, timestamp_str):
        """Captura screenshot com overlay mínimo"""
        frame_copy = frame.copy()
        
        # Adicionar texto de anomalia (vermelho, canto superior esquerdo)
        cv2.putText(frame_copy, "ANOMALIA DETECTADA", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Adicionar score (branco, menor)
        cv2.putText(frame_copy, f"Score: {combined_score:.4f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Adicionar timestamp (branco, canto inferior direito)
        cv2.putText(frame_copy, timestamp_str, (frame.shape[1]-200, frame.shape[0]-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame_copy

    def determine_risk_level(self, combined_score):
        """Determina nível de risco baseado no score"""
        threshold_base = max(self.cae_threshold, self.convlstm_threshold)
        
        if combined_score > threshold_base * 2.0:
            return "HIGH"
        elif combined_score > threshold_base * 1.5:
            return "MEDIUM"
        else:
            return "LOW"

    def generate_screenshot_filename(self, anomaly_type, risk_level, timestamp):
        """Gera nome do arquivo estruturado"""
        timestamp_str = timestamp.strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
        return f"{timestamp_str}_{anomaly_type}_{CAMERA_ID}_{risk_level}.jpg"

    def save_screenshot(self, frame, anomaly_type, combined_score, timestamp):
        """Salva screenshot com caminhos CORRETOS para a API"""
        risk_level = self.determine_risk_level(combined_score)
        filename = self.generate_screenshot_filename(anomaly_type, risk_level, timestamp)
        
        date_path_relative = f"{timestamp.year}/{timestamp.month:02d}/{timestamp.day:02d}"
        date_path_absolute = os.path.join(SCREENSHOTS_DIR, date_path_relative)
        os.makedirs(date_path_absolute, exist_ok=True)
        
        full_path = os.path.join(date_path_absolute, filename)
        
        timestamp_str = timestamp.strftime("%d/%m %H:%M:%S")
        screenshot_frame = self.capture_screenshot_with_overlay(
            frame, anomaly_type, combined_score, timestamp_str
        )
        
        success = cv2.imwrite(full_path, screenshot_frame)
        
        if success:
            print(f"📸 Screenshot salvo: {full_path}")
            return date_path_relative, filename, risk_level
        else:
            print(f"❌ Erro ao salvar screenshot: {filename}")
            return None, None, None

    def save_to_database(self, anomaly_data):
        """Salva anomalia na base de dados"""
        try:
            with sqlite3.connect(DB_NAME) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO anomalies_enhanced (
                        timestamp, type, location, cae_score, convlstm_score,
                        combined_score, screenshot_path, screenshot_filename,
                        camera_id, frame_resolution, risk_level,
                        detection_duration, frame_number, additional_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    anomaly_data['timestamp'],
                    anomaly_data['type'],
                    anomaly_data['location'],
                    anomaly_data['cae_score'],
                    anomaly_data['convlstm_score'],
                    anomaly_data['combined_score'],
                    anomaly_data['screenshot_path'],
                    anomaly_data['screenshot_filename'],
                    anomaly_data['camera_id'],
                    anomaly_data['frame_resolution'],
                    anomaly_data['risk_level'],
                    anomaly_data['detection_duration'],
                    anomaly_data['frame_number'],
                    json.dumps(anomaly_data['additional_data'])
                ))
                conn.commit()
            return True
        except Exception as e:
            print(f"❌ Erro ao salvar na BD: {e}")
            return False

    def send_notification_async(self, anomaly_data):
        """Envia notificação assíncrona"""
        def send():
            try:
                response = requests.post(
                    ANOMALY_LOGGING_API,
                    json=anomaly_data,
                    timeout=5
                )
                if response.status_code == 200:
                    print("✅ Notificação enviada com sucesso")
                else:
                    print(f"⚠️ Erro na notificação: {response.status_code}")
            except Exception as e:
                print(f"❌ Erro ao enviar notificação: {e}")
        
        thread = threading.Thread(target=send)
        thread.daemon = True
        thread.start()

    def process_anomaly(self, frame, anomaly_type, cae_score, convlstm_score, 
                       frame_number, detection_start_time):
        """Processa anomalia com caminhos corretos"""
        detection_end_time = time.time()
        detection_duration = detection_end_time - detection_start_time
        
        combined_score = max(cae_score, convlstm_score)
        timestamp = datetime.now()
        
        screenshot_path_relative, screenshot_filename, risk_level = self.save_screenshot(
            frame, anomaly_type, combined_score, timestamp
        )
        
        if screenshot_path_relative is None:
            return
        
        frame_resolution = f"{frame.shape[1]}x{frame.shape[0]}"
        
        anomaly_data = {
            'timestamp': timestamp.isoformat(),
            'type': anomaly_type,
            'location': DEFAULT_LOCATION,
            'cae_score': float(cae_score),
            'convlstm_score': float(convlstm_score),
            'combined_score': float(combined_score),
            'screenshot_path': screenshot_path_relative,
            'screenshot_filename': screenshot_filename,
            'camera_id': CAMERA_ID,
            'frame_resolution': frame_resolution,
            'risk_level': risk_level,
            'detection_duration': detection_duration,
            'frame_number': frame_number,
            'additional_data': {
                'thresholds_used': {
                    'cae': self.cae_threshold,
                    'convlstm': self.convlstm_threshold
                }
            }
        }
        
        self.save_to_database(anomaly_data)
        self.send_notification_async(anomaly_data)
        
        print(f"🚨 ANOMALIA PROCESSADA: {anomaly_type} | Score: {combined_score:.4f} | Risk: {risk_level}")
        print(f"📸 Screenshot: {screenshot_path_relative}/{screenshot_filename}")

    def run_detection(self, video_path):
        """✅ MELHORADO: Executa detecção com filtros anti-falso-positivo"""
        if not os.path.exists(video_path):
            print(f"❌ Vídeo não encontrado: {video_path}")
            return

        print(f"🎥 Iniciando detecção em: {video_path}")
        
        self.load_models()
        self.load_thresholds()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Erro ao abrir vídeo: {video_path}")
            return

        ret, prev_frame = cap.read()
        if not ret:
            print("❌ Erro ao ler primeiro frame")
            return

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.resize(prev_gray, (IMG_WIDTH, IMG_HEIGHT))
        prev_gray = self.preprocess_frame(prev_gray)  # ✅ Pré-processar
        
        flow_buffer = deque(maxlen=SEQUENCE_LENGTH)
        frame_count = 0
        frames_skipped = 0  # ⚡ Contador de frames pulados
        
        print("🔍 Detecção iniciada... Pressione 'q' para sair")
        print(f"📊 Configurações: MIN_MOVEMENT={MIN_MOVEMENT_THRESHOLD}, COOLDOWN={ANOMALY_COOLDOWN}s")
        
        while True:
            detection_start_time = time.time()
            ret, next_frame = cap.read()
            if not ret:
                break

            # ⚡ PULAR FRAMES para aumentar FPS (processar 1 a cada 3)
            frames_skipped += 1
            if frames_skipped % (SKIP_FRAMES + 1) != 0:
                frame_count += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # ⚡ REDIMENSIONAR frame se for muito grande
            if next_frame.shape[1] > MAX_FRAME_WIDTH:
                scale = MAX_FRAME_WIDTH / next_frame.shape[1]
                new_width = int(next_frame.shape[1] * scale)
                new_height = int(next_frame.shape[0] * scale)
                next_frame = cv2.resize(next_frame, (new_width, new_height))

            display_frame = next_frame.copy()
            next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
            next_gray = cv2.resize(next_gray, (IMG_WIDTH, IMG_HEIGHT))
            next_gray = self.preprocess_frame(next_gray)  # ✅ Pré-processar

            # ✅ MELHORADO: Calcular fluxo óptico com parâmetros mais robustos
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, next_gray, None,
                pyr_scale=0.5,
                levels=5,        # Aumentado
                winsize=21,      # Aumentado
                iterations=5,    # Aumentado
                poly_n=7,        # Aumentado
                poly_sigma=1.5,  # Aumentado
                flags=0
            )
            
            # ✅ NOVO: Verificar magnitude do movimento
            avg_magnitude, max_magnitude = self.calculate_flow_magnitude(flow)
            
            # ✅ NOVO: Ignorar frames com movimento muito pequeno (ruído)
            if avg_magnitude < MIN_MOVEMENT_THRESHOLD:
                prev_gray = next_gray
                frame_count += 1
                
                cv2.putText(display_frame, "Sem movimento significativo", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
                cv2.putText(display_frame, f"Magnitude: {avg_magnitude:.2f}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow('Detecção Melhorada', display_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            # ✅ NOVO: Aplicar máscara ROI (ignorar bordas)
            flow = self.apply_roi_mask(flow)
            
            # ✅ NOVO: Normalizar fluxo
            flow = cv2.normalize(flow, None, -1, 1, cv2.NORM_MINMAX)
            
            flow_buffer.append(flow)
            
            # CAE Score
            cae_input = np.expand_dims(flow, axis=0)
            reconstruction = self.cae_model.predict(cae_input, verbose=0)
            cae_score = np.mean(np.square(cae_input - reconstruction))
            cae_score = self.validate_score(cae_score)  # ✅ Validar
            
            # ConvLSTM Score
            convlstm_score = 0
            if len(flow_buffer) == SEQUENCE_LENGTH:
                sequence_for_convlstm = np.array(list(flow_buffer))[:-1]
                sequence_for_convlstm = np.expand_dims(sequence_for_convlstm, axis=0)
                
                target_flow = np.array(list(flow_buffer))[-1]
                target_flow = np.expand_dims(target_flow, axis=0)

                prediction = self.convlstm_model.predict(sequence_for_convlstm, verbose=0)
                convlstm_score = np.mean(np.square(target_flow - prediction))
                convlstm_score = self.validate_score(convlstm_score)  # ✅ Validar

            # ✅ MELHORADO: Detecção com SAFETY_FACTOR e lógica AND
            cae_exceeded = cae_score > (self.cae_threshold * SAFETY_FACTOR)
            convlstm_exceeded = convlstm_score > (self.convlstm_threshold * SAFETY_FACTOR)
            
            # Usar AND para ser mais rigoroso
            is_anomaly = cae_exceeded and convlstm_exceeded
            
            if is_anomaly:
                self.anomaly_counter += 1
                
                # ✅ NOVO: Requer confirmação em múltiplos frames
                if self.anomaly_counter >= ANOMALY_CONFIRMATION_FRAMES:
                    current_time = time.time()
                    
                    # ✅ NOVO: Cooldown entre alertas
                    if current_time - self.last_anomaly_time > ANOMALY_COOLDOWN:
                        anomaly_type = "INTRUSAO" if convlstm_score > cae_score else "MOVIMENTO_SUSPEITO"
                        
                        self.process_anomaly(
                            display_frame, anomaly_type, cae_score, convlstm_score,
                            frame_count, detection_start_time
                        )
                        
                        self.last_anomaly_time = current_time
                        self.anomaly_counter = 0
                
                cv2.putText(display_frame, f"ANOMALIA! (confirmando {self.anomaly_counter}/{ANOMALY_CONFIRMATION_FRAMES})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                self.anomaly_counter = 0  # Reset contador
                cv2.putText(display_frame, "Normal", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Mostrar informações
            fps = 1.0 / (time.time() - detection_start_time + 0.001)
            cv2.putText(display_frame, f"FPS: {fps:.1f}", 
                       (display_frame.shape[1]-100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(display_frame, f"CAE: {cae_score:.4f} (limiar: {self.cae_threshold*SAFETY_FACTOR:.4f})", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, f"ConvLSTM: {convlstm_score:.4f} (limiar: {self.convlstm_threshold*SAFETY_FACTOR:.4f})", 
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, f"Magnitude: {avg_magnitude:.2f}", 
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('Detecção Melhorada', display_frame)
            
            prev_gray = next_gray
            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("✅ Detecção concluída!")

def main():
    if len(sys.argv) != 2:
        print("Uso: python enhanced_detection_fixed.py <caminho_do_video>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    detector = EnhancedAnomalyDetector()
    detector.run_detection(video_path)

if __name__ == "__main__":
    main()
