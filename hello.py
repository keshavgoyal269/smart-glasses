# Smart Glasses AI — by Keshav Goyal
from flask import Flask, render_template, send_from_directory, request, jsonify
import os
import face_recognition
import numpy as np
import base64
from PIL import Image
import io

app = Flask(__name__)

# Load known faces on startup
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

        # Decode base64 image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        img_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_array = np.array(img)

        # Find faces
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

if __name__ == '__main__':
    app.run(debug=True)