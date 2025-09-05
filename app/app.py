import sqlite3
import os
import time
import json
from datetime import datetime, timedelta
from flask import Flask, jsonify, request, render_template, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- Configuração das Bases de Dados ---
DB_NAME = 'anomalies_enhanced.db'
SCREENSHOTS_DIR = '../screenshots'

def init_db():
    """
    Inicializa a base de dados expandida com todos os campos novos.
    """
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        
        # Tabela expandida com todos os novos campos
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
                additional_data TEXT,
                
                -- Campos para dashboard
                is_false_positive BOOLEAN DEFAULT 0,
                notes TEXT,
                reviewed BOOLEAN DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        ''')
        
        # Criar índices para performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON anomalies_enhanced(timestamp);')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_risk_level ON anomalies_enhanced(risk_level);')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_camera_id ON anomalies_enhanced(camera_id);')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_type ON anomalies_enhanced(type);')
        
        conn.commit()
    print("✅ Base de dados expandida inicializada.")

# Inicialização da BD
with app.app_context():
    init_db()

# --- Rotas do Servidor ---

@app.route('/')
def dashboard():
    """Serve a página principal do dashboard expandido."""
    return render_template('dashboard.html')

@app.route('/api/anomalies', methods=['GET'])
def get_anomalies():
    """
    API expandida com suporte a TODOS os filtros avançados do frontend
    """
    try:
        print("🔍 Buscando anomalias com filtros avançados...")
        
        # Parâmetros básicos
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 50))
        
        # 🔥 NOVOS FILTROS AVANÇADOS
        # Período
        hours_back = int(request.args.get('hours_back', 168))
        date_from = request.args.get('date_from')
        date_to = request.args.get('date_to')
        
        # Múltiplos níveis de risco (separados por vírgula)
        risk_levels = request.args.get('risk_level', '').split(',') if request.args.get('risk_level') else []
        risk_levels = [r.strip() for r in risk_levels if r.strip()]
        
        # Múltiplos tipos (separados por vírgula)
        anomaly_types = request.args.get('type', '').split(',') if request.args.get('type') else []
        anomaly_types = [t.strip() for t in anomaly_types if t.strip()]
        
        # Score range
        min_score = float(request.args.get('min_score', 0))
        max_score = float(request.args.get('max_score', 1))
        
        # Status filters
        include_false_positives = request.args.get('include_false_positives', 'true').lower() == 'true'
        include_reviewed = request.args.get('include_reviewed', 'true').lower() == 'true'
        include_unreviewed = request.args.get('include_unreviewed', 'true').lower() == 'true'
        
        # Pesquisa
        search_term = request.args.get('search', '').strip()
        
        print(f"📊 Filtros recebidos:")
        print(f"  - Período: {hours_back}h atrás")
        print(f"  - Risk levels: {risk_levels}")
        print(f"  - Types: {anomaly_types}")
        print(f"  - Score range: {min_score} - {max_score}")
        print(f"  - Pesquisa: '{search_term}'")
        
        # Calcular offset
        offset = (page - 1) * limit
        
        # 🔥 CONSTRUIR QUERY DINAMICAMENTE
        where_conditions = []
        params = []
        
        # Filtro de tempo
        if date_from and date_to:
            where_conditions.append("datetime(timestamp) BETWEEN ? AND ?")
            params.extend([date_from, date_to])
        else:
            time_threshold = datetime.now() - timedelta(hours=hours_back)
            where_conditions.append("datetime(timestamp) >= ?")
            params.append(time_threshold.isoformat())
        
        # Filtros de nível de risco (múltiplos)
        if risk_levels:
            placeholders = ','.join(['?' for _ in risk_levels])
            where_conditions.append(f"risk_level IN ({placeholders})")
            params.extend(risk_levels)
        
        # Filtros de tipo (múltiplos)
        if anomaly_types:
            placeholders = ','.join(['?' for _ in anomaly_types])
            where_conditions.append(f"type IN ({placeholders})")
            params.extend(anomaly_types)
        
        # Score range
        if min_score > 0 or max_score < 1:
            where_conditions.append("combined_score BETWEEN ? AND ?")
            params.extend([min_score, max_score])
        
        # Status filters
        status_conditions = []
        if include_false_positives:
            status_conditions.append("is_false_positive = 1")
        if include_reviewed:
            status_conditions.append("(reviewed = 1 AND (is_false_positive = 0 OR is_false_positive IS NULL))")
        if include_unreviewed:
            status_conditions.append("(reviewed = 0 OR reviewed IS NULL) AND (is_false_positive = 0 OR is_false_positive IS NULL)")
        
        if status_conditions:
            where_conditions.append(f"({' OR '.join(status_conditions)})")
        
        # Pesquisa em múltiplos campos
        if search_term:
            search_conditions = [
                "notes LIKE ?",
                "type LIKE ?", 
                "location LIKE ?"
            ]
            where_conditions.append(f"({' OR '.join(search_conditions)})")
            search_param = f"%{search_term}%"
            params.extend([search_param, search_param, search_param])
        
        where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
        
        print(f"🔎 Query WHERE: {where_clause}")
        print(f"🔎 Parâmetros: {params[:5]}...")  # Só primeiros 5 para não sobrecarregar log
        
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            
            # Query principal com paginação
            query = f"""
                SELECT * FROM anomalies_enhanced 
                {where_clause}
                ORDER BY datetime(timestamp) DESC 
                LIMIT ? OFFSET ?
            """
            params_with_pagination = params + [limit, offset]
            
            cursor.execute(query, params_with_pagination)
            anomalies = cursor.fetchall()
            
            # Query para contar total
            count_query = f"SELECT COUNT(*) FROM anomalies_enhanced {where_clause}"
            cursor.execute(count_query, params)
            total_count = cursor.fetchone()[0]
            
            print(f"📋 Anomalias encontradas: {len(anomalies)} de {total_count} total")
            
            # Obter nomes das colunas
            cursor.execute("PRAGMA table_info(anomalies_enhanced)")
            columns_info = cursor.fetchall()
            columns = [col[1] for col in columns_info]
            
            # Formatar resultados
            result = []
            for row in anomalies:
                try:
                    anomaly_dict = dict(zip(columns, row))
                    
                    # Garantir campos críticos
                    anomaly_dict['id'] = anomaly_dict.get('id', 0)
                    anomaly_dict['type'] = anomaly_dict.get('type', 'UNKNOWN')
                    anomaly_dict['combined_score'] = anomaly_dict.get('combined_score', 0.0)
                    anomaly_dict['risk_level'] = anomaly_dict.get('risk_level', 'MEDIUM')
                    anomaly_dict['timestamp'] = anomaly_dict.get('timestamp', datetime.now().isoformat())
                    anomaly_dict['location'] = anomaly_dict.get('location', 'Unknown')
                    anomaly_dict['camera_id'] = anomaly_dict.get('camera_id', 'CAM01')
                    anomaly_dict['is_false_positive'] = bool(anomaly_dict.get('is_false_positive', False))
                    anomaly_dict['reviewed'] = bool(anomaly_dict.get('reviewed', False))
                    
                    # Parse additional_data
                    if anomaly_dict.get('additional_data'):
                        try:
                            anomaly_dict['additional_data'] = json.loads(anomaly_dict['additional_data'])
                        except:
                            anomaly_dict['additional_data'] = {}
                    else:
                        anomaly_dict['additional_data'] = {}
                    
                    result.append(anomaly_dict)
                    
                except Exception as row_error:
                    print(f"❌ Erro ao processar linha: {row_error}")
                    continue
            
            # Calcular paginação
            total_pages = (total_count + limit - 1) // limit if total_count > 0 else 0
            
            # Resposta com metadata
            response = {
                'anomalies': result,
                'pagination': {
                    'page': page,
                    'limit': limit,
                    'total': total_count,
                    'pages': total_pages,
                    'has_next': offset + limit < total_count,
                    'has_prev': page > 1
                },
                'filters_applied': {
                    'hours_back': hours_back,
                    'risk_levels': risk_levels,
                    'types': anomaly_types,
                    'score_range': [min_score, max_score],
                    'search_term': search_term,
                    'total_found': total_count
                }
            }
            
            print(f"✅ Resposta preparada: {len(result)} anomalias")
            return jsonify(response)
            
    except Exception as e:
        print(f"❌ Erro na API get_anomalies: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            "error": f"Erro interno do servidor: {str(e)}",
            "details": "Verifique os logs para mais informações"
        }), 500


# @app.route('/api/anomalies', methods=['GET'])
# def get_anomalies():
#     """
#     API expandida para obter anomalias com filtros e paginação - CORRIGIDA.
#     """
#     try:
#         print("🔍 Buscando anomalias...")  # Debug
        
#         # Parâmetros de query
#         page = int(request.args.get('page', 1))
#         limit = int(request.args.get('limit', 50))
#         risk_level = request.args.get('risk_level', '')
#         camera_id = request.args.get('camera_id', '')
#         anomaly_type = request.args.get('type', '')
#         hours_back = int(request.args.get('hours_back', 168))  # 7 dias por padrão
#         include_false_positives = request.args.get('include_false_positives', 'true').lower() == 'true'
        
#         print(f"📊 Parâmetros: page={page}, limit={limit}, hours_back={hours_back}")
        
#         # Calcular offset
#         offset = (page - 1) * limit
        
#         # Construir query com filtros
#         where_conditions = []
#         params = []
        
#         # Filtro de tempo (mais permissivo para debug)
#         time_threshold = datetime.now() - timedelta(hours=hours_back)
#         where_conditions.append("datetime(timestamp) >= ?")
#         params.append(time_threshold.isoformat())
        
#         # Filtros opcionais
#         if risk_level:
#             where_conditions.append("risk_level = ?")
#             params.append(risk_level)
            
#         if camera_id:
#             where_conditions.append("camera_id = ?")
#             params.append(camera_id)
            
#         if anomaly_type:
#             where_conditions.append("type = ?")
#             params.append(anomaly_type)
            
#         if not include_false_positives:
#             where_conditions.append("(is_false_positive = 0 OR is_false_positive IS NULL)")
        
#         where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
        
#         print(f"🔎 Query WHERE: {where_clause}")
#         print(f"🔎 Parâmetros: {params}")
        
#         with sqlite3.connect(DB_NAME) as conn:
#             cursor = conn.cursor()
            
#             # Primeiro, verificar se a tabela existe e tem dados
#             cursor.execute("SELECT COUNT(*) FROM anomalies_enhanced")
#             total_records = cursor.fetchone()[0]
#             print(f"📈 Total de registos na BD: {total_records}")
            
#             if total_records == 0:
#                 # Retornar resposta vazia mas válida
#                 return jsonify({
#                     'anomalies': [],
#                     'pagination': {
#                         'page': page,
#                         'limit': limit,
#                         'total': 0,
#                         'pages': 0,
#                         'has_next': False,
#                         'has_prev': False
#                     }
#                 })
            
#             # Query principal com paginação
#             query = f"""
#                 SELECT * FROM anomalies_enhanced 
#                 {where_clause}
#                 ORDER BY id DESC 
#                 LIMIT ? OFFSET ?
#             """
#             params_with_pagination = params + [limit, offset]
            
#             print(f"🔎 Query completa: {query}")
#             print(f"🔎 Parâmetros finais: {params_with_pagination}")
            
#             cursor.execute(query, params_with_pagination)
#             anomalies = cursor.fetchall()
            
#             print(f"📋 Anomalias encontradas: {len(anomalies)}")
            
#             # Query para contar total
#             count_query = f"SELECT COUNT(*) FROM anomalies_enhanced {where_clause}"
#             cursor.execute(count_query, params)
#             total_count = cursor.fetchone()[0]
            
#             print(f"📊 Total com filtros: {total_count}")
            
#             # Obter nomes das colunas
#             cursor.execute("PRAGMA table_info(anomalies_enhanced)")
#             columns_info = cursor.fetchall()
#             columns = [col[1] for col in columns_info]  # col[1] é o nome da coluna
            
#             print(f"🏛️ Colunas da tabela: {columns}")
            
#             # Formatar resultados
#             result = []
            
#             for row in anomalies:
#                 try:
#                     anomaly_dict = dict(zip(columns, row))
                    
#                     # Garantir que campos críticos existem
#                     anomaly_dict['id'] = anomaly_dict.get('id', 0)
#                     anomaly_dict['type'] = anomaly_dict.get('type', 'UNKNOWN')
#                     anomaly_dict['combined_score'] = anomaly_dict.get('combined_score', 0.0)
#                     anomaly_dict['risk_level'] = anomaly_dict.get('risk_level', 'MEDIUM')
#                     anomaly_dict['timestamp'] = anomaly_dict.get('timestamp', datetime.now().isoformat())
#                     anomaly_dict['location'] = anomaly_dict.get('location', 'Unknown')
#                     anomaly_dict['camera_id'] = anomaly_dict.get('camera_id', 'CAM01')
#                     anomaly_dict['is_false_positive'] = bool(anomaly_dict.get('is_false_positive', False))
                    
#                     # Parse additional_data se existir
#                     if anomaly_dict.get('additional_data'):
#                         try:
#                             anomaly_dict['additional_data'] = json.loads(anomaly_dict['additional_data'])
#                         except:
#                             anomaly_dict['additional_data'] = {}
#                     else:
#                         anomaly_dict['additional_data'] = {}
                    
#                     result.append(anomaly_dict)
                    
#                 except Exception as row_error:
#                     print(f"❌ Erro ao processar linha: {row_error}")
#                     continue
            
#             # Calcular paginação
#             total_pages = (total_count + limit - 1) // limit if total_count > 0 else 0
            
#             # Resposta com metadata de paginação
#             response = {
#                 'anomalies': result,
#                 'pagination': {
#                     'page': page,
#                     'limit': limit,
#                     'total': total_count,
#                     'pages': total_pages,
#                     'has_next': offset + limit < total_count,
#                     'has_prev': page > 1
#                 }
#             }
            
#             print(f"✅ Resposta preparada: {len(result)} anomalias, {total_pages} páginas")
#             return jsonify(response)
            
#     except Exception as e:
#         print(f"❌ Erro na API get_anomalies: {e}")
#         import traceback
#         traceback.print_exc()
        
#         # Retornar erro mais detalhado
#         return jsonify({
#             "error": f"Erro interno do servidor: {str(e)}",
#             "details": "Verifique os logs do servidor para mais informações"
#         }), 500


@app.route('/api/log_anomaly', methods=['POST'])
def log_anomaly():
    """
    API expandida COMPATÍVEL com sistema antigo e novo.
    Aceita tanto 'score' quanto 'combined_score'
    """
    try:
        data = request.json
        print(f"📥 Dados recebidos: {data}")  # Debug
        
        # 🔥 COMPATIBILIDADE: Aceitar formato antigo OU novo
        if 'score' in data and 'combined_score' not in data:
            # Formato antigo - converter para novo
            combined_score = data['score']
            cae_score = data.get('cae_score', 0.0)
            convlstm_score = data.get('convlstm_score', combined_score)  # usar score como fallback
            print("🔄 Convertendo formato antigo para novo")
        else:
            # Formato novo
            combined_score = data.get('combined_score', data.get('score', 0.0))
            cae_score = data.get('cae_score', 0.0)
            convlstm_score = data.get('convlstm_score', 0.0)
            print("✅ Usando formato novo")
        
        # Validação básica (aceita ambos os formatos)
        if not data.get('type') or not combined_score or not data.get('location'):
            return jsonify({"error": "Campos obrigatórios: type, score/combined_score, location"}), 400

        # Extrair dados com valores padrão
        timestamp = data.get('timestamp', datetime.now().isoformat())
        anomaly_type = data['type']
        location = data['location']
        
        # Normalizar tipo de anomalia (corrigir encoding)
        anomaly_type = anomaly_type.replace('IntrusÃ£o', 'INTRUSAO').replace('Intrusão', 'INTRUSAO')
        
        screenshot_path = data.get('screenshot_path', '')
        screenshot_filename = data.get('screenshot_filename', '')
        camera_id = data.get('camera_id', 'CAM01')
        frame_resolution = data.get('frame_resolution', '')
        
        # Determinar risk_level se não fornecido
        if 'risk_level' not in data:
            if combined_score > 0.8:
                risk_level = 'HIGH'
            elif combined_score > 0.5:
                risk_level = 'MEDIUM'
            else:
                risk_level = 'LOW'
        else:
            risk_level = data['risk_level']
        
        detection_duration = data.get('detection_duration', 0.0)
        frame_number = data.get('frame_number', 0)
        additional_data = json.dumps(data.get('additional_data', {}))
        
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
                timestamp, anomaly_type, location, cae_score, convlstm_score,
                combined_score, screenshot_path, screenshot_filename,
                camera_id, frame_resolution, risk_level, detection_duration,
                frame_number, additional_data
            ))
            anomaly_id = cursor.lastrowid
            conn.commit()

        print(f"✅ Anomalia registada: ID={anomaly_id}, Tipo={anomaly_type}, Score={combined_score}, Risk={risk_level}")
        return jsonify({
            "message": "Anomalia registada com sucesso!",
            "anomaly_id": anomaly_id
        }), 201

    except Exception as e:
        print(f"❌ Erro ao processar anomalia: {e}")
        import traceback
        traceback.print_exc()  # Debug completo
        return jsonify({"error": "Erro interno do servidor"}), 500

@app.route('/api/anomalies/<int:anomaly_id>/mark_false_positive', methods=['POST'])
def mark_false_positive(anomaly_id):
    """Marca uma anomalia como falso positivo."""
    try:
        data = request.json or {}
        notes = data.get('notes', '')
        
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE anomalies_enhanced 
                SET is_false_positive = 1, reviewed = 1, notes = ?
                WHERE id = ?
            ''', (notes, anomaly_id))
            
            if cursor.rowcount == 0:
                return jsonify({"error": "Anomalia não encontrada"}), 404
                
            conn.commit()
        
        return jsonify({"message": "Marcado como falso positivo"}), 200
        
    except Exception as e:
        print(f"Erro ao marcar falso positivo: {e}")
        return jsonify({"error": "Erro interno"}), 500

@app.route('/api/anomalies/<int:anomaly_id>/add_note', methods=['POST'])
def add_note(anomaly_id):
    """Adiciona nota a uma anomalia."""
    try:
        data = request.json
        notes = data.get('notes', '')
        
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE anomalies_enhanced 
                SET notes = ?, reviewed = 1
                WHERE id = ?
            ''', (notes, anomaly_id))
            
            if cursor.rowcount == 0:
                return jsonify({"error": "Anomalia não encontrada"}), 404
                
            conn.commit()
        
        return jsonify({"message": "Nota adicionada"}), 200
        
    except Exception as e:
        print(f"Erro ao adicionar nota: {e}")
        return jsonify({"error": "Erro interno"}), 500

# @app.route('/api/screenshot/<path:filename>')
# def serve_screenshot(filename):
#     """Serve screenshots para o dashboard."""
#     try:
#         file_path = os.path.join(SCREENSHOTS_DIR, filename)
#         if os.path.exists(file_path):
#             return send_file(file_path)
#         else:
#             return jsonify({"error": "Screenshot não encontrado"}), 404
#     except Exception as e:
#         print(f"Erro ao servir screenshot: {e}")
#         return jsonify({"error": "Erro interno"}), 500

# Substituir esta função no app.py

@app.route('/api/screenshot/<path:filename>')
def serve_screenshot(filename):
    """Serve screenshots para o dashboard - CORRIGIDA."""
    try:
        print(f"📸 Pedido de screenshot: {filename}")
        
        # O filename pode vir como "2025/01/15/arquivo.jpg" ou só "arquivo.jpg"
        full_path = os.path.join(SCREENSHOTS_DIR, filename)
        
        print(f"🔍 Procurando em: {full_path}")
        print(f"📁 SCREENSHOTS_DIR: {SCREENSHOTS_DIR}")
        print(f"📁 Caminho existe? {os.path.exists(full_path)}")
        
        if os.path.exists(full_path) and os.path.isfile(full_path):
            print(f"✅ Screenshot encontrado: {full_path}")
            return send_file(full_path, mimetype='image/jpeg')
        else:
            # Tentar procurar recursivamente (fallback)
            print(f"🔍 Procurando recursivamente por: {os.path.basename(filename)}")
            
            for root, dirs, files in os.walk(SCREENSHOTS_DIR):
                if os.path.basename(filename) in files:
                    fallback_path = os.path.join(root, os.path.basename(filename))
                    print(f"✅ Screenshot encontrado (fallback): {fallback_path}")
                    return send_file(fallback_path, mimetype='image/jpeg')
            
            print(f"❌ Screenshot não encontrado: {filename}")
            # Retornar imagem placeholder ou erro 404
            return jsonify({"error": "Screenshot não encontrado"}), 404
            
    except Exception as e:
        print(f"❌ Erro ao servir screenshot: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Erro interno: {str(e)}"}), 500

# Adicionar endpoint de debug para screenshots
@app.route('/api/debug/screenshots')
def debug_screenshots():
    """Debug: Listar todos os screenshots disponíveis."""
    try:
        screenshots = []
        
        if os.path.exists(SCREENSHOTS_DIR):
            for root, dirs, files in os.walk(SCREENSHOTS_DIR):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        full_path = os.path.join(root, file)
                        relative_path = os.path.relpath(full_path, SCREENSHOTS_DIR)
                        screenshots.append({
                            'filename': file,
                            'relative_path': relative_path,
                            'full_path': full_path,
                            'size': os.path.getsize(full_path),
                            'exists': os.path.exists(full_path)
                        })
        
        return jsonify({
            'screenshots_dir': SCREENSHOTS_DIR,
            'screenshots_dir_exists': os.path.exists(SCREENSHOTS_DIR),
            'total_screenshots': len(screenshots),
            'screenshots': screenshots[:10]  # Primeiros 10 para não sobrecarregar
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def get_stats():
    """API para estatísticas do dashboard."""
    try:
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            
            # Estatísticas das últimas 24h
            last_24h = datetime.now() - timedelta(hours=24)
            
            # Total anomalias hoje
            cursor.execute('''
                SELECT COUNT(*) FROM anomalies_enhanced 
                WHERE datetime(timestamp) >= ? AND is_false_positive = 0
            ''', (last_24h.isoformat(),))
            total_24h = cursor.fetchone()[0]
            
            # Por nível de risco
            cursor.execute('''
                SELECT risk_level, COUNT(*) FROM anomalies_enhanced 
                WHERE datetime(timestamp) >= ? AND is_false_positive = 0
                GROUP BY risk_level
            ''', (last_24h.isoformat(),))
            by_risk = dict(cursor.fetchall())
            
            # Por tipo
            cursor.execute('''
                SELECT type, COUNT(*) FROM anomalies_enhanced 
                WHERE datetime(timestamp) >= ? AND is_false_positive = 0
                GROUP BY type
            ''', (last_24h.isoformat(),))
            by_type = dict(cursor.fetchall())
            
            # Por câmara
            cursor.execute('''
                SELECT camera_id, COUNT(*) FROM anomalies_enhanced 
                WHERE datetime(timestamp) >= ? AND is_false_positive = 0
                GROUP BY camera_id
            ''', (last_24h.isoformat(),))
            by_camera = dict(cursor.fetchall())
            
            # Últimas anomalias por hora (para gráfico)
            cursor.execute('''
                SELECT 
                    strftime('%H', timestamp) as hour,
                    COUNT(*) as count
                FROM anomalies_enhanced 
                WHERE datetime(timestamp) >= ? AND is_false_positive = 0
                GROUP BY strftime('%H', timestamp)
                ORDER BY hour
            ''', (last_24h.isoformat(),))
            hourly_data = dict(cursor.fetchall())
            
            stats = {
                'total_24h': total_24h,
                'high_risk': by_risk.get('HIGH', 0),
                'medium_risk': by_risk.get('MEDIUM', 0),
                'low_risk': by_risk.get('LOW', 0),
                'by_type': by_type,
                'by_camera': by_camera,
                'hourly_data': hourly_data,
                'system_status': 'online'
            }
            
            return jsonify(stats)
            
    except Exception as e:
        print(f"Erro ao obter estatísticas: {e}")
        return jsonify({"error": "Erro interno"}), 500

# Adicionar este endpoint ao app.py para debug

@app.route('/api/debug/db')
def debug_database():
    """Endpoint de debug para verificar estado da base de dados."""
    try:
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            
            # Verificar se tabela existe
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='anomalies_enhanced'")
            table_exists = cursor.fetchone() is not None
            
            # Contar registos
            cursor.execute("SELECT COUNT(*) FROM anomalies_enhanced")
            total_count = cursor.fetchone()[0]
            
            # Obter estrutura da tabela
            cursor.execute("PRAGMA table_info(anomalies_enhanced)")
            columns = cursor.fetchall()
            
            # Últimos 3 registos
            cursor.execute("SELECT * FROM anomalies_enhanced ORDER BY id DESC LIMIT 3")
            recent_records = cursor.fetchall()
            
            debug_info = {
                'database_file': DB_NAME,
                'database_exists': os.path.exists(DB_NAME),
                'table_exists': table_exists,
                'total_records': total_count,
                'columns': [{'name': col[1], 'type': col[2]} for col in columns],
                'recent_records': len(recent_records),
                'sample_data': [dict(zip([col[1] for col in columns], record)) for record in recent_records]
            }
            
            return jsonify(debug_info)
            
    except Exception as e:
        return jsonify({
            'error': str(e),
            'database_file': DB_NAME,
            'database_exists': os.path.exists(DB_NAME)
        }), 500
    
if __name__ == '__main__':
    print("🚀 API Flask Expandida iniciada!")
    print("📊 Dashboard: http://localhost:5000")
    print("🔌 API: http://localhost:5000/api/")
    app.run(debug=True)