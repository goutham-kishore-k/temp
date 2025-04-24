from flask import Flask, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io
import os
#from flask_ngrok import run_with_ngrok

app = Flask(__name__)
#run_with_ngrok(app)

model = load_model("model_v3_v1.h5", compile=False)
class_labels = ['Angry', 'Happy', 'Sad', 'Other']

def prepare_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

@app.route("/")
def serve_html():
    return send_from_directory(".", "index.html")

@app.route('/predict', methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "no file uploaded"}), 400
    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read()))
    prepared = prepare_image(image)
    prediction = model.predict(prepared)
    result = class_labels[np.argmax(prediction)]
    return jsonify({"emotion": result})

if __name__ == "__main__":
    app.run(debug=True)