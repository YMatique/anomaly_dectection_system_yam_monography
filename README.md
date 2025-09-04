# anomaly_dectection_system_yam_monography
System designed for anomaly detection in suvigillance video

# ACTIVATE ENVIRONMENT: 
 python -m venv tf_env   
 .\tf_env\Scripts\activate

# ALGUMAS INSTRUÇÕES
Ao treinar certificar que o diretório de prparação de dados : INPUT_VIDEOS_DIR = '../videos_input' esteja apontando aos dados de treino.
E ao gerar dados de teste também certificar que esteja rodando com o caminho certo para teste.
# PACKAGES
## open cv: pip install opencv-python numpy  
## tensorflow: pip install tensorflow  (Enable long paths in Windows 10, version 1607, and later: open powershell as admin: New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force)
## mathplot: pip install matplotlib
## Flask: pip install Flask 
## Flask Cors: pip install flask_cors




# PARA USAR
## abra primeiro o app/app.py: cd app/app.py
## depois abra outro terminal e rode o arquivo anomaly_detection.py <caminho-do-video.extensao>

## depois podes testar a api: http://127.0.0.1:5000