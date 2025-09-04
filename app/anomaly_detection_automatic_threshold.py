import os
import sys
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from collections import deque
import time

# --- Dependências Necessárias ---
# Certifique-se de que tem estas bibliotecas instaladas:
# pip install tensorflow opencv-python matplotlib

# --- Configurações e Parâmetros ---
# Pasta onde os modelos treinados estão salvos.
MODELS_DIR = '../trained_models'
# Pasta onde os dados de Fluxo Óptico 'normal' foram salvos. Usamos isto para determinar o limiar.
OPTICAL_FLOW_NORMAL_DIR = '../optical_flow_data/normal'
# OPTICAL_FLOW_NORMAL_DIR = '../optical_flow_data/treino'
# Pasta para salvar os screenshots das anomalias.
SCREENSHOTS_DIR = '../screenshots'

# Caminho para o arquivo que armazena os limiares calculados.
THRESHOLDS_FILE = os.path.join(MODELS_DIR, 'anomaly_thresholds.txt')

# Dimensões dos frames de Fluxo Óptico
IMG_HEIGHT, IMG_WIDTH = 64, 64
IMG_CHANNELS = 2

# Tamanho da sequência de frames para o ConvLSTM
SEQUENCE_LENGTH = 10 

# --- Funções de Ajuda ---

def load_trained_models():
    """
    Carrega os modelos CAE e ConvLSTM treinados a partir do diretório de modelos.
    """
    print("A carregar modelos treinados...")
    try:
        cae_model = load_model(os.path.join(MODELS_DIR, 'cae_final.keras'))
        convlstm_model = load_model(os.path.join(MODELS_DIR, 'convlstm_final.keras'))
        return cae_model, convlstm_model
    except Exception as e:
        print(f"Erro: Não foi possível carregar os modelos. Certifique-se de que o treinamento foi concluído. Erro: {e}")
        sys.exit(1)

def calculate_cae_threshold(cae_model):
    """
    Calcula o limiar de anomalia para o CAE com base nos dados de treino.
    """
    print("A calcular limiar do CAE a partir dos dados de treino...")
    all_normal_flows = []
    for video_dir in sorted(os.listdir(OPTICAL_FLOW_NORMAL_DIR)):
        video_path = os.path.join(OPTICAL_FLOW_NORMAL_DIR, video_dir)
        if os.path.isdir(video_path):
            for flow_file in sorted(os.listdir(video_path)):
                if flow_file.endswith('.npy'):
                    flow_frame = np.load(os.path.join(video_path, flow_file))
                    all_normal_flows.append(flow_frame)
    
    if not all_normal_flows:
        print("Erro: Nenhum dado de fluxo óptico 'normal' encontrado para calcular o limiar.")
        sys.exit(1)
        
        
    all_normal_flows = np.array(all_normal_flows)
    
    # Calcular as pontuações de reconstrução com o CAE
    reconstructions = cae_model.predict(all_normal_flows, verbose=0)
    mse = np.mean(np.square(all_normal_flows - reconstructions), axis=(1, 2, 3))
    
    # Limiar = Média + 2 desvios padrão
    threshold = np.mean(mse) + 2 * np.std(mse)
    return threshold

def calculate_convlstm_threshold(convlstm_model):
    """
    Calcula o limiar de anomalia para o ConvLSTM com base nos dados de treino.
    """
    print("A calcular limiar do ConvLSTM a partir dos dados de treino...")
    all_sequences = []
    for video_dir in sorted(os.listdir(OPTICAL_FLOW_NORMAL_DIR)):
        video_path = os.path.join(OPTICAL_FLOW_NORMAL_DIR, video_dir)
        if os.path.isdir(video_path):
            video_flows = []
            for flow_file in sorted(os.listdir(video_path)):
                if flow_file.endswith('.npy'):
                    flow_frame = np.load(os.path.join(video_path, flow_file))
                    video_flows.append(flow_frame)
            
            if len(video_flows) >= SEQUENCE_LENGTH:
                video_flows = np.array(video_flows)
                for i in range(len(video_flows) - SEQUENCE_LENGTH + 1):
                    all_sequences.append(video_flows[i:i + SEQUENCE_LENGTH])
    
    if not all_sequences:
        print("Erro: Nenhuma sequência de dados 'normal' encontrada para calcular o limiar.")
        sys.exit(1)

    all_sequences = np.array(all_sequences)
    
    input_data = all_sequences[:, :-1]
    target_data = all_sequences[:, -1]
    
    predictions = convlstm_model.predict(input_data, verbose=0)
    mse = np.mean(np.square(target_data - predictions), axis=(1, 2, 3))
    
    threshold = np.mean(mse) + 2 * np.std(mse)
    return threshold

def save_thresholds(cae_threshold, convlstm_threshold):
    """
    Salva os limiares calculados num arquivo de texto para uso futuro.
    """
    with open(THRESHOLDS_FILE, 'w') as f:
        f.write(f"cae_threshold:{cae_threshold}\n")
        f.write(f"convlstm_threshold:{convlstm_threshold}\n")
    print(f"Limiares salvos em: {THRESHOLDS_FILE}")

def load_thresholds():
    """
    Tenta carregar os limiares a partir do arquivo. Retorna os valores se o arquivo
    existir, ou None se não existir.
    """
    if os.path.exists(THRESHOLDS_FILE):
        print("Limiares pré-calculados encontrados. A carregar...")
        try:
            with open(THRESHOLDS_FILE, 'r') as f:
                lines = f.readlines()
                cae_threshold = float(lines[0].split(':')[1])
                convlstm_threshold = float(lines[1].split(':')[1])
            return cae_threshold, convlstm_threshold
        except Exception as e:
            print(f"Erro ao carregar os limiares, a recalcular. Erro: {e}")
            return None, None
    return None, None

def main(video_path):
    """
    Função principal para carregar modelos, processar um vídeo e detetar anomalias.
    """
    if not os.path.exists(video_path):
        print(f"Erro: O arquivo de vídeo '{video_path}' não foi encontrado.")
        sys.exit(1)

    # Carregar os modelos treinados
    cae_model, convlstm_model = load_trained_models()
    
    # Tenta carregar os limiares de um arquivo para otimizar o processo.
    cae_threshold, convlstm_threshold = load_thresholds()

    # Se os limiares não foram encontrados, calcula-os e depois salva-os.
    if cae_threshold is None or convlstm_threshold is None:
        cae_threshold = calculate_cae_threshold(cae_model)
        convlstm_threshold = calculate_convlstm_threshold(convlstm_model)
        save_thresholds(cae_threshold, convlstm_threshold)

    # Criar a pasta para screenshots se não existir
    if not os.path.exists(SCREENSHOTS_DIR):
        os.makedirs(SCREENSHOTS_DIR)
    
    # Abrir o vídeo para processamento
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erro: Não foi possível abrir o vídeo em '{video_path}'.")
        sys.exit(1)

    # Inicializar variáveis
    ret, prev_frame = cap.read()
    if not ret:
        print("Erro: Não foi possível ler o primeiro frame do vídeo.")
        sys.exit(1)

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.resize(prev_gray, (IMG_WIDTH, IMG_HEIGHT))
    
    flow_buffer = deque(maxlen=SEQUENCE_LENGTH)
    cae_scores = []
    convlstm_scores = []
    
    print(f"\nA processar o vídeo '{video_path}'...")
    print("Pressione 's' para tirar um screenshot.")
    print("Pressione 'q' para sair.")
    frame_count = 0
    while True:
        ret, next_frame = cap.read()
        if not ret:
            break

        # Aumentar a escala da frame para visualização
        display_frame = next_frame.copy()

        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.resize(next_gray, (IMG_WIDTH, IMG_HEIGHT))

        # Calcular o fluxo óptico Farneback
        flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # O fluxo óptico tem 2 canais (x e y), o que corresponde ao nosso IMG_CHANNELS
        flow_buffer.append(flow)
        
        # CAE Score: calcular para cada frame
        cae_input = np.expand_dims(flow, axis=0)
        reconstruction = cae_model.predict(cae_input, verbose=0)
        cae_score = np.mean(np.square(cae_input - reconstruction))
        cae_scores.append(cae_score)
        
        # ConvLSTM Score: calcular quando o buffer tiver sequências suficientes
        if len(flow_buffer) == SEQUENCE_LENGTH:
            sequence_for_convlstm = np.array(list(flow_buffer))[:-1] # Os primeiros N-1 frames
            sequence_for_convlstm = np.expand_dims(sequence_for_convlstm, axis=0) # Adicionar dimensão do batch
            
            target_flow = np.array(list(flow_buffer))[-1] # O último frame é o alvo
            target_flow = np.expand_dims(target_flow, axis=0)

            prediction = convlstm_model.predict(sequence_for_convlstm, verbose=0)
            convlstm_score = np.mean(np.square(target_flow - prediction))
            convlstm_scores.append(convlstm_score)
            
            # --- Visualização em tempo real ---
            # O ConvLSTM tem um atraso de N frames, então usamos a última pontuação para o frame atual
            is_anomaly = (cae_score > cae_threshold) or (convlstm_score > convlstm_threshold)

            text_display = "Normal"
            color = (0, 255, 0)  # Verde
            
            if is_anomaly:
                text_display = "ANOMALIA DETECTADA!"
                color = (0, 0, 255)  # Vermelho
            
            cv2.putText(display_frame, text_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        prev_gray = next_gray
        frame_count += 1

        cv2.imshow('Deteccao de Anomalias em Tempo Real', display_frame)

        key = cv2.waitKey(1) & 0xFF
        
        # Se a tecla 'q' for pressionada, sair do loop
        if key == ord('q'):
            break
        # Se a tecla 's' for pressionada, tirar um screenshot
        elif key == ord('s'):
            timestamp = int(time.time())
            screenshot_filename = os.path.join(SCREENSHOTS_DIR, f"screenshot_{timestamp}.png")
            cv2.imwrite(screenshot_filename, display_frame)
            print(f"Screenshot salvo em: {screenshot_filename}")

    cap.release()
    cv2.destroyAllWindows()
    
    print("Processamento do vídeo concluído.")
    
    # --- Plotar os resultados ---
    # Plot para o CAE
    plt.figure(figsize=(15, 6))
    plt.plot(cae_scores, label='Pontuação de Anomalia (CAE)', color='blue')
    plt.hlines(cae_threshold, 0, len(cae_scores), colors='red', linestyles='dashed', label='Limiar de Anomalia')
    plt.title('Pontuação de Anomalia (CAE)')
    plt.xlabel('Frame')
    plt.ylabel('Pontuação (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(MODELS_DIR, 'live_cae_anomaly_scores.png'))
    plt.close()

    # Plot para o ConvLSTM
    plt.figure(figsize=(15, 6))
    plt.plot(convlstm_scores, label='Pontuação de Anomalia (ConvLSTM)', color='green')
    plt.hlines(convlstm_threshold, 0, len(convlstm_scores), colors='red', linestyles='dashed', label='Limiar de Anomalia')
    plt.title('Pontuação de Anomalia (ConvLSTM)')
    plt.xlabel('Frame')
    plt.ylabel('Pontuação (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(MODELS_DIR, 'live_convlstm_anomaly_scores.png'))
    plt.close()
    
    print("Gráficos de pontuação de anomalia salvos na pasta 'trained_models'.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python live_anomaly_detection.py <caminho_do_video>")
        sys.exit(1)
    
    video_path_arg = sys.argv[1]
    main(video_path_arg)
