import cv2
import numpy as np
import os
import sys

# --- Dependências Necessárias ---
# Certifique-se de que tem estas bibliotecas instaladas:
# pip install opencv-python numpy

# --- Configurações ---
# Caminho para a pasta que contém os vídeos de entrada.
# Deve seguir a estrutura: videos_input/[normal ou anomalo]/[seu_video].mp4
INPUT_VIDEOS_DIR = '../videos_input'
# Caminho para a pasta onde o Fluxo Óptico será salvo.
OUTPUT_OPTICAL_FLOW_DIR = '../optical_flow_data'
# Resolução para redimensionar os frames antes de calcular o Fluxo Óptico.
# Uma resolução menor acelera o processamento.
# RESIZE_DIM = (128, 128) # Exemplo: (largura, altura)
RESIZE_DIM = (64, 64)

# --- Funções Auxiliares ---

def visualize_optical_flow(flow):
    """
    Visualiza o Fluxo Óptico usando um mapa de cores HSV.
    A cor representa a direção e a intensidade representa a magnitude do movimento.
    """
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255 # Saturação total

    # Calcula a magnitude e o ângulo dos vetores de movimento
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Mapeia o ângulo para o matiz (cor) e a magnitude para o valor (brilho)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def process_video_for_optical_flow(video_path, output_dir, resize_dim=None):
    """
    Processa um único vídeo, calcula o Fluxo Óptico e salva os frames resultantes.

    Args:
        video_path (str): Caminho completo para o arquivo de vídeo.
        output_dir (str): Diretório onde os dados do Fluxo Óptico serão salvos.
        resize_dim (tuple, optional): Dimensões para redimensionar os frames (largura, altura).
                                      Se None, não redimensiona.
    Returns:
        bool: True se o processamento foi bem-sucedido, False caso contrário.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erro: Não foi possível abrir o vídeo {video_path}")
        return False

    # Cria o diretório de saída para este vídeo
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)
    print(f"Processando vídeo: {video_path} -> Salvando em: {video_output_dir}")

    ret, prev_frame = cap.read()
    if not ret:
        print(f"Erro: Não foi possível ler o primeiro frame de {video_path}")
        cap.release()
        return False

    if resize_dim:
        prev_frame = cv2.resize(prev_frame, resize_dim)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if resize_dim:
            frame = cv2.resize(frame, resize_dim)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calcula o Fluxo Óptico usando o algoritmo Farnebäck
        flow = cv2.calcOpticalFlowFarneback(
            prev=prev_gray,
            next=gray,
            flow=None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        # Normaliza o Fluxo Óptico para o intervalo [-1, 1] ou [0, 1]
        # para ser adequado como entrada para redes neurais.
        normalized_flow = flow / (np.max(np.abs(flow)) + 1e-8)

        # Salva o array numpy do Fluxo Óptico
        np.save(os.path.join(video_output_dir, f'flow_{frame_idx:05d}.npy'), normalized_flow)

        # --- Visualização (Opcional, para depuração) ---
        # Remova ou comente as linhas abaixo se não precisar de visualização durante o processamento
        # flow_vis = visualize_optical_flow(flow)
        # cv2.imshow('Original Frame', frame)
        # cv2.imshow('Optical Flow Visualization', flow_vis)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        # --- Fim da Visualização ---

        prev_gray = gray
        frame_idx += 1

    cap.release()
    # cv2.destroyAllWindows() # Descomente se usou a visualização
    print(f"Processamento de {video_path} concluído. {frame_idx} frames de Fluxo Óptico salvos.")
    return True

# --- Função Principal ---

def main():
    """
    Função principal para processar todos os vídeos nas subpastas.
    """
    if not os.path.exists(INPUT_VIDEOS_DIR):
        print(f"Erro: A pasta de vídeos de entrada '{INPUT_VIDEOS_DIR}' não existe.")
        print("Por favor, crie-a e coloque os seus vídeos lá, organizados em subpastas 'normal' e 'anomalo'.")
        sys.exit(1)

    os.makedirs(OUTPUT_OPTICAL_FLOW_DIR, exist_ok=True)

    for category in os.listdir(INPUT_VIDEOS_DIR):
        category_path = os.path.join(INPUT_VIDEOS_DIR, category)
        if os.path.isdir(category_path):
            output_category_dir = os.path.join(OUTPUT_OPTICAL_FLOW_DIR, category)
            os.makedirs(output_category_dir, exist_ok=True)
            print(f"\nProcessando categoria: {category}")

            # for video_file in os.listdir(category_path):
            #     if video_file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            #         video_path = os.path.join(category_path, video_file)
            #         process_video_for_optical_flow(video_path, output_category_dir, RESIZE_DIM)
            #     else:
            #         print(f"Ignorando arquivo não-vídeo: {video_file}")
            for root, _, files in os.walk(category_path):  # percorre recursivamente
                for video_file in files:
                    if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                        video_path = os.path.join(root, video_file)

                        # mantém a estrutura relativa no output
                        relative_path = os.path.relpath(root, category_path)
                        final_output_dir = os.path.join(output_category_dir, relative_path)
                        os.makedirs(final_output_dir, exist_ok=True)

                        process_video_for_optical_flow(video_path, final_output_dir, RESIZE_DIM)
                    else:
                        print(f"Ignorando arquivo não-vídeo: {video_file}")

    print("\nProcessamento de Fluxo Óptico concluído para todos os vídeos.")
    print(f"Os dados foram salvos em: {OUTPUT_OPTICAL_FLOW_DIR}")

if __name__ == "__main__":
    main()
