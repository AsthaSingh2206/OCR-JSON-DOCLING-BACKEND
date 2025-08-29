# flask_api.py
from flask import Flask, request, jsonify
import sqlite3
import os
import tempfile
import json
from colab_ocr_pipeline import process_pdf
from docling.datamodel.accelerator_options import AcceleratorDevice

app = Flask(__name__)
DB_PATH = 'ocr_docs.db'

@app.route('/docs', methods=['GET'])
def list_docs():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS docs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        data JSON
    )''')
    c.execute('SELECT id, filename FROM docs')
    docs = [{'id': row[0], 'filename': row[1]} for row in c.fetchall()]
    conn.close()
    return jsonify(docs)

@app.route('/query', methods=['POST'])
def query_json():
    data = request.get_json(force=True)
    sql = data.get('sql')
    if not sql:
        return jsonify({'error': 'Missing SQL query'}), 400
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    try:
        c.execute(sql)
        rows = [dict(row) for row in c.fetchall()]
        conn.close()
        return jsonify(rows)
    except Exception as e:
        conn.close()
        return jsonify({'error': str(e)}), 400

@app.route('/doc/<int:doc_id>', methods=['GET'])
def get_doc(doc_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS docs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        data JSON
    )''')
    c.execute('SELECT data FROM docs WHERE id=?', (doc_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return jsonify(json.loads(row[0]))
    else:
        return jsonify({'error': 'Document not found'}), 404

@app.route('/upload', methods=['POST'])
def upload_doc():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    ocr_backend = request.form.get('ocr_backend', 'rapid')
    accelerator_device = request.form.get('accelerator_device', 'CPU')
    try:
        device_enum = AcceleratorDevice[accelerator_device.upper()]
    except Exception:
        device_enum = AcceleratorDevice.CPU
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name
    try:
        out_path, json_result = process_pdf(tmp_path, os.path.dirname(tmp_path), ocr_backend, device_enum)
        os.remove(tmp_path)
        # Store in DB
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS docs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            data JSON
        )''')
        c.execute('INSERT INTO docs (filename, data) VALUES (?, ?)', (file.filename, json.dumps(json_result)))
        conn.commit()
        conn.close()
        return jsonify(json_result)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
