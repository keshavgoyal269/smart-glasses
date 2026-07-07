# Smart Glasses AI — by Keshav Goyal
from flask import Flask, render_template, send_from_directory, request, jsonify
import os
import face_recognition
import numpy as np
import base64
from PIL import Image
import io
import urllib.request
import json
from ultralytics import YOLO
from ddgs import DDGS

app = Flask(__name__)

# Load YOLO model
yolo_model = YOLO('yolov8n.pt')

known_encodings = []
known_names = []

def load_known_faces():
    global known_encodings, known_names
    known_encodings = []
    known_names = []
    faces_dir = 'faces'
    if not os.path.exists(faces_dir):
        return
    for filename in os.listdir(faces_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(faces_dir, filename)
            try:
                img = face_recognition.load_image_file(path)
                encs = face_recognition.face_encodings(img)
                if encs:
                    known_encodings.append(encs[0])
                    name = os.path.splitext(filename)[0].replace('_', ' ').title()
                    known_names.append(name)
                    print(f'Loaded face: {name}')
            except Exception as e:
                print(f'Error loading {filename}: {e}')
    print(f'Total known faces: {len(known_names)}')

load_known_faces()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/faces/<filename>')
def serve_face(filename):
    return send_from_directory('faces', filename)

@app.route('/faces-list')
def faces_list():
    faces = os.listdir('faces')
    faces = [f for f in faces if f.endswith(('.jpg', '.jpeg', '.png'))]
    return {'faces': faces}

@app.route('/recognize', methods=['POST'])
def recognize():
    try:
        data = request.get_json()
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        img_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_array = np.array(img)
        face_locations = face_recognition.face_locations(img_array)
        face_encs = face_recognition.face_encodings(img_array, face_locations)
        results = []
        for enc in face_encs:
            name = 'Unknown'
            confidence = 0
            if known_encodings:
                distances = face_recognition.face_distance(known_encodings, enc)
                best_idx = np.argmin(distances)
                best_dist = distances[best_idx]
                if best_dist < 0.55:
                    name = known_names[best_idx]
                    confidence = round((1 - best_dist) * 100)
            results.append({'name': name, 'confidence': confidence})
        return jsonify({'faces': results})
    except Exception as e:
        print(f'Recognition error: {e}')
        return jsonify({'faces': [], 'error': str(e)})

def is_realtime_question(question):
    keywords = ['who is', 'current', 'president', 'prime minister', 'ceo', 'latest',
                'today', 'news', 'weather', 'price', 'stock', 'score', 'winner',
                'right now', 'currently', 'recent', '2024', '2025', '2026']
    q = question.lower()
    return any(k in q for k in keywords)

def web_search(query):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query + " 2026", max_results=5))
            if results:
                for r in results:
                    body = r['body']
                    if any(name in body for name in ['Trump', 'Biden', 'Modi', 'president', 'minister']):
                        return body[:300]
                return results[0]['body'][:300]
    except Exception as e:
        print(f'Web search error: {e}')
    return None

def ask_ollama(prompt):
    payload = json.dumps({
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    }).encode('utf-8')
    req = urllib.request.Request(
        'http://localhost:11434/api/generate',
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode('utf-8'))
        return result['response']

@app.route('/ask', methods=['POST'])
def ask_ai():
    try:
        data = request.get_json()
        question = data.get('question', '')

        if is_realtime_question(question):
            print(f'Using web search for: {question}')
            web_answer = web_search(question)
            if web_answer:
                prompt = f"Extract and state the direct answer from this text in one sentence. Text: '{web_answer}'. Question: {question}. Just state the fact directly, no disclaimers."
                answer = ask_ollama(prompt)
                print(f'Web+AI answer: {answer}')
                return jsonify({'answer': answer})

        print(f'Using Ollama for: {question}')
        prompt = f"You are JEEVIKA, a smart glasses AI assistant. Answer briefly in 2-3 sentences only. Question: {question}"
        answer = ask_ollama(prompt)
        print(f'AI answer: {answer}')
        return jsonify({'answer': answer})

    except Exception as e:
        print(f'AI error: {e}')
        return jsonify({'answer': 'I could not get an answer right now. Please try again.'})

if __name__ == '__main__':
    app.run(debug=True)