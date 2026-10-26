import re, joblib
from flask import Flask, request, jsonify, send_from_directory
from features import extract, reasons, normalize
from urllib.parse import urlparse

app = Flask(__name__, static_folder='static')
model = joblib.load('model.joblib')

@app.get('/')
def home():
    return send_from_directory('static', 'index.html')

@app.post('/predict')
def predict():
    text = (request.get_json(silent=True) or {}).get('text', '')
    m = re.search(r'(?:https?://|www\.)[^\s<>"\']+', text, re.I) or re.search(r'\b[a-z0-9-]+(?:\.[a-z0-9-]+)+(?:/\S*)?', text, re.I)
    if not m:
        return jsonify(error='No link found'), 400
    url = m.group(0).rstrip('.,;:!?)')
    p = float(model.predict_proba([extract(url)])[0][1])
    level = 'safe' if p < 0.35 else 'warn' if p < 0.65 else 'bad'
    host = (urlparse(normalize(url)).hostname or url)
    return jsonify(level=level, risk=round(p * 100), host=host, reasons=reasons(url))

if __name__ == '__main__':
    app.run(debug=True)
