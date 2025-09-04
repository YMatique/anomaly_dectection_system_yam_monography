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
SEQUENCE_LENGTH = 10 

# API do servidor
ANOMALY_LOGGING_API = 'http://127.0.0.1:5000/api/log_anomaly'

# Configurações de screenshot
CAMERA_ID = 'CAM01'
DEFAULT_LOCATION = 'Entrada Principal'

class EnhancedAnomalyDetector:
    def __init__(self):
        self.cae_model = None
        self.convlstm_model = None
        self.cae_threshold = None
        self.convlstm_threshold = None
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
        """Calcula limiares baseados nos dados de treino"""
        print("🧮 Calculando limiares...")
        # Implementação simplificada - você pode usar a do código original
        self.cae_threshold = 0.05  # Valor exemplo
        self.convlstm_threshold = 0.08  # Valor exemplo
        
        # Salvar para próxima execução
        with open(THRESHOLDS_FILE, 'w') as f:
            f.write(f"cae_threshold:{self.cae_threshold}\n")
            f.write(f"convlstm_threshold:{self.convlstm_threshold}\n")

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
        if combined_score > self.convlstm_threshold * 1.5:
            return "HIGH"
        elif combined_score > self.convlstm_threshold * 1.2:
            return "MEDIUM"
        else:
            return "LOW"

    def generate_screenshot_filename(self, anomaly_type, risk_level, timestamp):
        """Gera nome do arquivo estruturado"""
        timestamp_str = timestamp.strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]  # milissegundos
        return f"{timestamp_str}_{anomaly_type}_{CAMERA_ID}_{risk_level}.jpg"

    def save_screenshot(self, frame, anomaly_type, combined_score, timestamp):
        """Salva screenshot com caminhos CORRETOS para a API"""
        # Determinar nível de risco
        risk_level = self.determine_risk_level(combined_score)
        
        # Gerar nome do arquivo
        filename = self.generate_screenshot_filename(anomaly_type, risk_level, timestamp)
        
        # CAMINHO RELATIVO PARA API (importante!)
        date_path_relative = f"{timestamp.year}/{timestamp.month:02d}/{timestamp.day:02d}"
        
        # CAMINHO ABSOLUTO PARA SALVAR
        date_path_absolute = os.path.join(SCREENSHOTS_DIR, date_path_relative)
        os.makedirs(date_path_absolute, exist_ok=True)
        
        # Caminho completo do arquivo
        full_path = os.path.join(date_path_absolute, filename)
        
        # Capturar com overlay
        timestamp_str = timestamp.strftime("%d/%m %H:%M:%S")
        screenshot_frame = self.capture_screenshot_with_overlay(
            frame, anomaly_type, combined_score, timestamp_str
        )
        
        # Salvar
        success = cv2.imwrite(full_path, screenshot_frame)
        
        if success:
            print(f"📸 Screenshot salvo: {full_path}")
            # RETORNAR CAMINHO RELATIVO PARA A API
            return date_path_relative, filename, risk_level
        else:
            print(f"❌ Erro ao salvar screenshot: {filename}")
            return None, None, None
    def save_to_database(self, anomaly_data):
        """Salva anomalia na base de dados expandida"""
        try:
            with sqlite3.connect(DB_NAME) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO anomalies_enhanced (
                        timestamp, type, location, cae_score, convlstm_score, 
                        combined_score, screenshot_path, screenshot_filename,
                        camera_id, frame_resolution, risk_level, detection_duration,
                        frame_number, additional_data
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
                    json.dumps(anomaly_data.get('additional_data', {}))
                ))
                conn.commit()
                print(f"💾 Anomalia salva na BD: {anomaly_data['type']}")
                return True
        except Exception as e:
            print(f"❌ Erro ao salvar na BD: {e}")
            return False

    def send_notification_async(self, anomaly_data):
        """Envia notificação de forma assíncrona"""
        def send_notification():
            try:
                # Enviar para API original (compatibilidade)
                # api_data = {
                #     'type': anomaly_data['type'],
                #     'score': anomaly_data['combined_score'],
                #     'location': anomaly_data['location']
                # }
                api_data = {
                    'timestamp': anomaly_data['timestamp'],
                    'type': anomaly_data['type'],
                    'location': anomaly_data['location'],
                    'cae_score': anomaly_data['cae_score'],
                    'convlstm_score': anomaly_data['convlstm_score'],
                    'combined_score': anomaly_data['combined_score'],
                    # 📸 INCLUIR DADOS DE SCREENSHOT
                    'screenshot_path': anomaly_data['screenshot_path'],
                    'screenshot_filename': anomaly_data['screenshot_filename'],
                    'camera_id': anomaly_data['camera_id'],
                    'frame_resolution': anomaly_data['frame_resolution'],
                    'risk_level': anomaly_data['risk_level'],
                    'detection_duration': anomaly_data['detection_duration'],
                    'frame_number': anomaly_data['frame_number'],
                    'additional_data': anomaly_data['additional_data']
                }
                response = requests.post(ANOMALY_LOGGING_API, json=api_data, timeout=5)
                
                if response.status_code == 201:
                    print("📡 Notificação enviada com sucesso")
                else:
                    print(f"⚠️ Erro na notificação: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ Erro ao enviar notificação: {e}")
        
        # Executar em thread separada para não bloquear
        thread = threading.Thread(target=send_notification)
        thread.daemon = True
        thread.start()

    # def process_anomaly(self, frame, anomaly_type, cae_score, convlstm_score, 
    #                    frame_number, detection_start_time):
    #     """Processa uma anomalia detectada - screenshot, BD, notificação"""
    #     detection_end_time = time.time()
    #     detection_duration = detection_end_time - detection_start_time
        
    #     # Score combinado (pode ajustar a lógica)
    #     combined_score = max(cae_score, convlstm_score)
        
    #     # Timestamp atual
    #     timestamp = datetime.now()
        
    #     # 1. Salvar screenshot
    #     screenshot_path, screenshot_filename, risk_level = self.save_screenshot(
    #         frame, anomaly_type, combined_score, timestamp
    #     )
        
    #     if screenshot_path is None:
    #         return  # Erro ao salvar screenshot
        
    #     # 2. Preparar dados para BD
    #     frame_resolution = f"{frame.shape[1]}x{frame.shape[0]}"
        
    #     anomaly_data = {
    #         'timestamp': timestamp.isoformat(),
    #         'type': anomaly_type,
    #         'location': DEFAULT_LOCATION,
    #         'cae_score': float(cae_score),
    #         'convlstm_score': float(convlstm_score),
    #         'combined_score': float(combined_score),
    #         'screenshot_path': screenshot_path,
    #         'screenshot_filename': screenshot_filename,
    #         'camera_id': CAMERA_ID,
    #         'frame_resolution': frame_resolution,
    #         'risk_level': risk_level,
    #         'detection_duration': detection_duration,
    #         'frame_number': frame_number,
    #         'additional_data': {
    #             'thresholds_used': {
    #                 'cae': self.cae_threshold,
    #                 'convlstm': self.convlstm_threshold
    #             }
    #         }
    #     }
        
    #     # 3. Salvar na BD
    #     success = self.save_to_database(anomaly_data)
        
    #     # 4. Enviar notificação (assíncrona)
    #     if success:
    #         self.send_notification_async(anomaly_data)
        
    #     print(f"🚨 ANOMALIA PROCESSADA: {anomaly_type} | Score: {combined_score:.4f} | Risk: {risk_level}")

    def process_anomaly(self, frame, anomaly_type, cae_score, convlstm_score, 
                   frame_number, detection_start_time):
        """Processa anomalia com caminhos corretos"""
        detection_end_time = time.time()
        detection_duration = detection_end_time - detection_start_time
        
        combined_score = max(cae_score, convlstm_score)
        timestamp = datetime.now()
        
        # 1. Salvar screenshot
        screenshot_path_relative, screenshot_filename, risk_level = self.save_screenshot(
            frame, anomaly_type, combined_score, timestamp
        )
        
        if screenshot_path_relative is None:
            return  # Erro ao salvar screenshot
        
        # 2. Preparar dados para BD
        frame_resolution = f"{frame.shape[1]}x{frame.shape[0]}"
        
        anomaly_data = {
            'timestamp': timestamp.isoformat(),
            'type': anomaly_type,
            'location': DEFAULT_LOCATION,
            'cae_score': float(cae_score),
            'convlstm_score': float(convlstm_score),
            'combined_score': float(combined_score),
            'screenshot_path': screenshot_path_relative,  # ← CAMINHO RELATIVO
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
        
        # 3. Salvar na BD
        # success = self.save_to_database(anomaly_data)
        
        # 4. Enviar notificação (assíncrona)
        # if success:
        #     self.send_notification_async(anomaly_data)
        self.send_notification_async(anomaly_data)
        print(f"🚨 ANOMALIA PROCESSADA: {anomaly_type} | Score: {combined_score:.4f} | Risk: {risk_level}")
        print(f"📸 Screenshot: {screenshot_path_relative}/{screenshot_filename}")
    def run_detection(self, video_path):
        """Executa detecção com sistema de registo melhorado"""
        if not os.path.exists(video_path):
            print(f"❌ Vídeo não encontrado: {video_path}")
            return

        print(f"🎥 Iniciando detecção em: {video_path}")
        
        # Carregar modelos e limiares
        self.load_models()
        self.load_thresholds()
        
        # Abrir vídeo
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Erro ao abrir vídeo: {video_path}")
            return

        # Inicializar variáveis
        ret, prev_frame = cap.read()
        if not ret:
            print("❌ Erro ao ler primeiro frame")
            return

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.resize(prev_gray, (IMG_WIDTH, IMG_HEIGHT))
        
        flow_buffer = deque(maxlen=SEQUENCE_LENGTH)
        frame_count = 0
        
        print("🔍 Detecção iniciada... Pressione 'q' para sair")
        
        while True:
            detection_start_time = time.time()
            ret, next_frame = cap.read()
            if not ret:
                break

            display_frame = next_frame.copy()
            next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
            next_gray = cv2.resize(next_gray, (IMG_WIDTH, IMG_HEIGHT))

            # Calcular fluxo óptico
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            flow_buffer.append(flow)
            
            # CAE Score
            cae_input = np.expand_dims(flow, axis=0)
            reconstruction = self.cae_model.predict(cae_input, verbose=0)
            cae_score = np.mean(np.square(cae_input - reconstruction))
            
            # ConvLSTM Score
            convlstm_score = 0
            if len(flow_buffer) == SEQUENCE_LENGTH:
                sequence_for_convlstm = np.array(list(flow_buffer))[:-1]
                sequence_for_convlstm = np.expand_dims(sequence_for_convlstm, axis=0)
                
                target_flow = np.array(list(flow_buffer))[-1]
                target_flow = np.expand_dims(target_flow, axis=0)

                prediction = self.convlstm_model.predict(sequence_for_convlstm, verbose=0)
                convlstm_score = np.mean(np.square(target_flow - prediction))

            # Detectar anomalia
            is_anomaly = (cae_score > self.cae_threshold) or (convlstm_score > self.convlstm_threshold)
            
            if is_anomaly:
                anomaly_type = "INTRUSAO" if convlstm_score > cae_score else "MOVIMENTO_SUSPEITO"
                
                # 🔥 PROCESSAR ANOMALIA COMPLETA
                self.process_anomaly(
                    display_frame, anomaly_type, cae_score, convlstm_score,
                    frame_count, detection_start_time
                )
                
                # Feedback visual
                cv2.putText(display_frame, "ANOMALIA DETECTADA!", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(display_frame, "Normal", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Mostrar scores
            cv2.putText(display_frame, f"CAE: {cae_score:.4f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, f"ConvLSTM: {convlstm_score:.4f}", (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

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
        print("Uso: python enhanced_detection.py <caminho_do_video>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    detector = EnhancedAnomalyDetector()
    detector.run_detection(video_path)

if __name__ == "__main__":
    main()