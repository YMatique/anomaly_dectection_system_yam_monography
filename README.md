# Sistema de Detecção de Anomalias em Vigilância Residencial

## 📋 Descrição do Projeto

Este projeto implementa um **sistema híbrido automatizado** para detecção de anomalias comportamentais em vídeos de vigilância residencial, utilizando técnicas avançadas de **Deep Learning** e **Visão Computacional**. O sistema é capaz de identificar automaticamente atividades suspeitas e comportamentos anômalos em tempo real.

### 🎯 Características Principais

- **Detecção em Tempo Real**: Processamento de vídeo a 25-30 FPS
- **Arquitetura Híbrida**: Combina Optical Flow + CAE + ConvLSTM
- **Interface Web**: Dashboard interativo para monitorização
- **Baixo Custo Computacional**: Otimizado para hardware residencial
- **Sistema de Alertas**: Notificações automáticas e screenshots
- **Aprendizado Não Supervisionado**: Treina apenas com dados normais

## 🏗️ Arquitetura do Sistema

O sistema utiliza uma abordagem de **processamento em cascata** integrando três tecnologias:

```
📹 Vídeo → 🔄 Optical Flow → 🧠 CAE → 🔄 ConvLSTM → ⚠️ Detecção
```

### Componentes Principais:

1. **Optical Flow (Farnebäck)**: Detecção inicial rápida de movimento
2. **CAE (Convolutional Autoencoder)**: Análise espacial de padrões anômalos
3. **ConvLSTM**: Análise temporal de sequências comportamentais
4. **Dashboard Web**: Interface para visualização e controle

## 🛠️ Tecnologias Utilizadas

### Backend & AI
- **Python 3.8+** - Linguagem principal
- **TensorFlow 2.10+** - Framework de Deep Learning
- **OpenCV 4.6+** - Processamento de vídeo e Optical Flow
- **NumPy** - Computação científica
- **Flask 2.0+** - API e servidor web

### Frontend & Interface
- **HTML5/CSS3** - Interface web
- **JavaScript** - Interatividade
- **Chart.js** - Gráficos e visualizações
- **Tailwind CSS** - Estilização

### Base de Dados
- **SQLite** - Armazenamento de anomalias
- **Pandas** - Manipulação de dados

## 📁 Estrutura do Projeto

```
anomaly_detection_system/
├── 📁 app/                          # Aplicação principal
│   ├── 📄 app.py                    # Servidor Flask e APIs
│   ├── 📄 anomaly_detection.py      # Sistema de detecção melhorado
│   ├── 📄 model_training.py         # Treinamento dos modelos
│   ├── 📄 data_preparation.py       # Preparação de dados
│   └── 📁 templates/
│       └── 📄 dashboard.html        # Interface web principal
├── 📁 trained_models/               # Modelos treinados
│   ├── 📄 cae_final.keras          # Modelo CAE
│   ├── 📄 convlstm_final.keras     # Modelo ConvLSTM
│   └── 📄 anomaly_thresholds.txt   # Limiares de detecção
├── 📁 optical_flow_data/           # Dados processados
├── 📁 videos_input/                # Vídeos de entrada
├── 📁 screenshots/                 # Screenshots de anomalias
├── 📁 diagrams/                    # Diagramas da arquitetura
└── 📄 README.md                    # Este arquivo
```

## ⚙️ Requisitos do Sistema

### Hardware Mínimo
- **CPU**: Intel i5 ou equivalente (4+ cores)
- **RAM**: 8 GB (recomendado 16 GB)
- **Armazenamento**: 5 GB livres
- **GPU**: Opcional (acelera treinamento)

### Software
- **SO**: Windows 10/11, Linux Ubuntu 18.04+
- **Python**: 3.8 ou superior
- **Webcam/Câmera IP**: Para captura de vídeo

## 🚀 Instalação e Configuração

### 1. Configurar Ambiente Virtual

```bash
# Criar ambiente virtual
python -m venv tf_env

# Ativar ambiente (Windows)
.\tf_env\Scripts\activate

# Ativar ambiente (Linux/Mac)
source tf_env/bin/activate
```

### 2. Instalar Dependências

```bash
# Instalar pacotes essenciais
pip install tensorflow>=2.10
pip install opencv-python
pip install flask flask-cors
pip install numpy pandas matplotlib
pip install scikit-learn requests
```

### 3. Preparar Estrutura de Dados

```bash
# Criar diretórios necessários
mkdir videos_input videos_input/treino videos_input/teste
mkdir optical_flow_data trained_models screenshots
```

## 📊 Como Usar o Sistema

### 1. Preparação dos Dados

Organize seus vídeos de treinamento:
```
videos_input/
├── treino/          # Vídeos de comportamento normal
│   ├── normal_01.mp4
│   └── normal_02.mp4
└── teste/           # Vídeos para validação
    ├── normal_test.mp4
    └── anomaly_test.mp4
```

### 2. Processar Vídeos

```bash
cd app
python data_preparation.py
```

### 3. Treinar Modelos

```bash
python model_training.py
```

**Parâmetros de Treinamento:**
- **Épocas**: 30
- **Learning Rate**: 0.001
- **Batch Size**: 4
- **Sequência**: 10 frames
- **Resolução**: 64x64 pixels

### 4. Executar Sistema de Detecção

```bash
# Terminal 1: Iniciar servidor web
python app.py

# Terminal 2: Executar detecção em vídeo
python anomaly_detection.py <caminho_do_video.mp4>
```

### 5. Acessar Dashboard

Abra o navegador em: **http://localhost:5000**

## 📈 Funcionalidades do Dashboard

### 🎛️ Painel Principal
- **Estatísticas em tempo real** (24h)
- **Gráficos de anomalias por hora**
- **Distribuição por tipo de anomalia**
- **Filtros avançados** (risco, tipo, período)

### 📊 Gestão de Anomalias
- **Feed de anomalias recentes**
- **Screenshots automáticos**
- **Sistema de notas e comentários**
- **Marcação de falsos positivos**
- **Paginação e busca**

### ⚙️ APIs Disponíveis

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/api/anomalies` | GET | Lista anomalias com filtros |
| `/api/log_anomaly` | POST | Registra nova anomalia |
| `/api/stats` | GET | Estatísticas do sistema |
| `/api/screenshot/<path>` | GET | Serve imagens de anomalias |

## 🎯 Tipos de Anomalias Detectadas

### 🚨 Anomalias de Segurança
- **Intrusões não autorizadas**
- **Tentativas de arrombamento**
- **Movimentos suspeitos noturnos**
- **Escalada de muros/janelas**
- **Presença prolongada de estranhos**

### 🏥 Anomalias de Emergência
- **Quedas (especialmente idosos)**
- **Desmaios ou colapsos**
- **Imobilidade prolongada**
- **Movimentos erráticos/anômalos**

## 📊 Performance e Métricas

### Resultados Obtidos
- **Precisão CAE**: 72.16%
- **Precisão ConvLSTM**: 69.45%
- **Precisão Média**: 70.81%
- **Taxa de Processamento**: 25-30 FPS
- **Tempo de Resposta**: <300ms
- **Uso de CPU**: 50-60%
- **Uso de RAM**: 6-8 GB

### Especificações dos Modelos
| Modelo | Parâmetros | Tamanho | Tempo Treinamento |
|--------|------------|---------|-------------------|
| CAE | 12,770 | 49.88 KB | ~5 horas |
| ConvLSTM | 263,618 | 1.01 MB | ~31 horas |

## 🔧 Solução de Problemas

### Problemas Comuns

**1. Erro de GPU/CUDA**
```bash
# Verificar se TensorFlow reconhece GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Forçar uso de CPU se necessário
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

**2. Erro de LongPaths (Windows)**
```powershell
# Executar como administrador
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
                 -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

**3. Problemas de Performance**
- Reduzir resolução de processamento
- Diminuir batch size
- Verificar uso de RAM

### Debug do Sistema

```bash
# Verificar estado da base de dados
curl http://localhost:5000/api/debug/db

# Listar screenshots disponíveis
curl http://localhost:5000/api/debug/screenshots

# Testar API de anomalias
curl http://localhost:5000/api/anomalies?limit=5
```

## 🤝 Contribuição

Para contribuir com o projeto:

1. **Fork** o repositório
2. Crie uma **branch** para sua feature (`git checkout -b feature/nova-funcionalidade`)
3. **Commit** suas mudanças (`git commit -m 'Adiciona nova funcionalidade'`)
4. **Push** para a branch (`git push origin feature/nova-funcionalidade`)
5. Abra um **Pull Request**

## 📜 Licença

Este projeto é desenvolvido como parte de trabalho acadêmico (Monografia) para fins educacionais e de pesquisa.

## 👨‍💻 Autor

**Yuvi Matique** - Trabalho de Conclusão de Curso
- Universidade: [Sua Universidade]
- Curso: [Seu Curso]
- Ano: 2025

## 📚 Referências e Trabalhos Relacionados

- Hasan et al. (2016) - *Learning temporal regularity in video sequences*
- Shi et al. (2015) - *Convolutional LSTM network*
- Luo et al. (2017) - *Remembering history with convolutional LSTM*
- Farnebäck (2003) - *Two-frame motion estimation*

## 🔮 Desenvolvimentos Futuros

- [ ] **Suporte multi-câmera** para vigilância completa
- [ ] **Detecção de objetos específicos** (armas, pacotes)
- [ ] **Integração com IoT** residencial
- [ ] **App móvel** para alertas
- [ ] **Reconhecimento facial** de moradores
- [ ] **Análise de sentimentos** comportamentais
- [ ] **Detecção de som** complementar

---

**⚡ Sistema pronto para proteger sua residência com Inteligência Artificial!**


---

## ** Atualização **
# Câmera padrão (webcam)
python enhanced_detection.py

# Ou explicitamente
python enhanced_detection.py camera

# Câmera específica por ID
python enhanced_detection.py 0
python enhanced_detection.py 1

# Arquivo de vídeo (mantém compatibilidade)
python enhanced_detection.py caminho/do/video.mp4