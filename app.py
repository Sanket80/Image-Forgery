from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = load_model('MobileNetV2_more_dense_layer.keras')

# Path to store uploaded images temporarily
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Class names based on your model's training
class_names = ['real', 'fake']  # Adjust as per your model's classes

# Preprocess the image to the required input size
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))  # Change to 128x128
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalizing
    return img_array

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(request.url)

    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Preprocess and make prediction
        img_array = preprocess_image(file_path)
        prediction = model.predict(img_array)
        y_pred_class = np.argmax(prediction, axis=1)[0]  # Get predicted class
        confidence = np.amax(prediction) * 100  # Get confidence percentage
        result = class_names[y_pred_class]  # Get class name based on predicted class

        return render_template('result.html', result=result, confidence=confidence, image_path=file_path)

if __name__ == "__main__":
    app.run(debug=True)
