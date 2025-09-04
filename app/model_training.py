import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, ConvLSTM2D, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# --- Dependências Necessárias ---
# Certifique-se de que tem estas bibliotecas instaladas:
# pip install tensorflow matplotlib

# --- Configurações e Parâmetros ---
# Pasta onde os dados de Fluxo Óptico foram salvos.
OPTICAL_FLOW_DATA_DIR = '../optical_flow_data'
# Caminho para a pasta onde os modelos treinados e os gráficos serão salvos.
MODELS_DIR = '../trained_models'
os.makedirs(MODELS_DIR, exist_ok=True)

# Dimensões dos frames de Fluxo Óptico (ajuste se necessário)
# Deve ser as mesmas dimensões usadas no script de pré-processamento.
# IMG_HEIGHT, IMG_WIDTH = 128, 128  #ANES
IMG_HEIGHT, IMG_WIDTH = 64, 64
# O Fluxo Óptico tem 2 canais (x e y).
IMG_CHANNELS = 2

# Parâmetros de treinamento 
# EPOCHS = 50
EPOCHS = 30
# BATCH_SIZE = 8
BATCH_SIZE = 4
SEQUENCE_LENGTH = 10 # Tamanho da sequência de frames para o ConvLSTM

# --- Funções de Ajuda ---

def load_optical_flow_data(data_dir, sequence_length):
    """
    Carrega dados de Fluxo Óptico e os prepara em sequências.
    A função só carrega dados da subpasta 'normal' para treinamento.
    """
    print("A carregar dados de Fluxo Óptico...")
    normal_data_path = os.path.join(data_dir, 'treino')
        # normal_data_path = os.path.join(data_dir, 'normal')
    if not os.path.exists(normal_data_path):
        print(f"Erro: Pasta '{normal_data_path}' não encontrada.")
        return None

    sequences = []
    # Itera sobre os diretórios de vídeos individuais dentro da pasta 'normal'.
    for video_dir in sorted(os.listdir(normal_data_path)):
        video_path = os.path.join(normal_data_path, video_dir)
        if not os.path.isdir(video_path):
            continue

        video_flows = []
        # Carrega os frames de Fluxo Óptico de um vídeo.
        for flow_file in sorted(os.listdir(video_path)):
            if flow_file.endswith('.npy'):
                flow_frame = np.load(os.path.join(video_path, flow_file))
                video_flows.append(flow_frame)

        if len(video_flows) >= sequence_length:
            video_flows = np.array(video_flows)
            # Cria sequências de frames para o ConvLSTM
            for i in range(len(video_flows) - sequence_length + 1):
                sequences.append(video_flows[i:i + sequence_length])

    sequences = np.array(sequences)
    if sequences.size == 0:
        print("Nenhuma sequência de dados carregada. Verifique os seus dados de entrada.")
        return None
    print(f"Dados carregados: {sequences.shape[0]} sequências de {sequence_length} frames cada.")
    return sequences

def build_cae_model(input_shape):
    """
    Constrói e compila o modelo Convolutional Autoencoder (CAE).
    """
    input_layer = Input(shape=input_shape)

    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(IMG_CHANNELS, (3, 3), activation='sigmoid', padding='same')(x)

    model = Model(input_layer, decoded)
    # Usamos o MSE como função de perda para medir a qualidade da reconstrução.
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()
    return model

def build_convlstm_model(input_shape):
    """
    Constrói e compila o modelo ConvLSTM.
    O objetivo é prever o próximo frame de uma sequência.
    """
    # A entrada para o ConvLSTM é uma sequência de frames, então a forma é
    # (sequence_length, height, width, channels)
    convlstm_input = Input(shape=input_shape)
    
    # Camada ConvLSTM para capturar dependências espaciais e temporais
    x = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True, activation='relu')(convlstm_input)
    x = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=False, activation='relu')(x)
    
    # Camadas de saída para prever o próximo frame
    x = Conv2D(filters=IMG_CHANNELS, kernel_size=(3, 3), padding='same', activation='tanh')(x)
    
    # A saída do modelo é a previsão do próximo frame, por isso a forma da saída
    # é (batch, height, width, channels)
    model = Model(inputs=convlstm_input, outputs=x)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()
    return model

def plot_training_history(history, model_name, save_path):
    """
    Plota e salva o gráfico de perda do treinamento.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Perda de Treinamento')
    plt.title(f'Curva de Perda do Treinamento para o Modelo {model_name}')
    plt.xlabel('Época')
    plt.ylabel('Perda (Loss)')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    print(f"Gráfico de treinamento salvo em: {save_path}")
    plt.close()


def main():
    """
    Função principal para carregar dados, construir e treinar os modelos.
    """
    # 1. Carregar e preparar os dados
    sequences = load_optical_flow_data(OPTICAL_FLOW_DATA_DIR, SEQUENCE_LENGTH)
    if sequences is None:
        return

    # Separar os dados para treinamento do CAE e ConvLSTM
    # O CAE precisa de frames individuais, o ConvLSTM precisa de sequências.
    cae_train_data = sequences.reshape(-1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    # Para o ConvLSTM, a entrada são os primeiros N-1 frames e a saída é o último frame
    convlstm_train_data = sequences[:, :-1]
    convlstm_target_data = sequences[:, -1]

    # 2. Construir e treinar o modelo CAE
    print("\n--- A treinar o CAE ---")
    cae_input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    cae_model = build_cae_model(cae_input_shape)
    cae_callbacks = [EarlyStopping(monitor='loss', patience=5),
                     ModelCheckpoint(filepath=os.path.join(MODELS_DIR, 'cae_model_best_.keras'),
                                     save_best_only=True)]
    
    cae_history = cae_model.fit(cae_train_data, cae_train_data,
                                epochs=EPOCHS,
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                callbacks=cae_callbacks)
    
    # Salvar o modelo final e o gráfico
    cae_model.save(os.path.join(MODELS_DIR, 'cae_final.keras'))
    plot_training_history(cae_history, 'CAE', os.path.join(MODELS_DIR, 'cae_training_loss.png'))
    print(f"Modelo CAE salvo em {os.path.join(MODELS_DIR, 'cae_final.keras')}")


    # 3. Construir e treinar o modelo ConvLSTM
    print("\n--- A treinar o ConvLSTM ---")
    convlstm_input_shape = (SEQUENCE_LENGTH - 1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    convlstm_model = build_convlstm_model(convlstm_input_shape)
    convlstm_callbacks = [EarlyStopping(monitor='loss', patience=5),
                          ModelCheckpoint(filepath=os.path.join(MODELS_DIR, 'convlstm_model_best_.keras'),
                                          save_best_only=True)]
    
    convlstm_history = convlstm_model.fit(convlstm_train_data, convlstm_target_data,
                                          epochs=EPOCHS,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True,
                                          callbacks=convlstm_callbacks)
    
    # Salvar o modelo final e o gráfico
    convlstm_model.save(os.path.join(MODELS_DIR, 'convlstm_final.keras'))
    plot_training_history(convlstm_history, 'ConvLSTM', os.path.join(MODELS_DIR, 'convlstm_training_loss.png'))
    print(f"Modelo ConvLSTM salvo em {os.path.join(MODELS_DIR, 'convlstm_final.keras')}")

if __name__ == "__main__":
    main()
