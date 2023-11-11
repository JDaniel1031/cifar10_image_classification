from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import tensorflow as tf
import os

app = Flask(__name__)

# Load your trained model
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'models/cifar10_model.h5')

model = tf.keras.models.load_model(model_path)
class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def preprocess_image(image_url):
    # Download the image
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    img = img.resize((32, 32))  # Adjust the size as per your model requirements
    img_array = np.array(img) / 255.0  # Normalize the pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_class(image_array):
    # Use your trained model to predict the class
    prediction = model.predict(image_array)
    class_idx = np.argmax(prediction)
    return class_idx

def get_class_name(class_idx):
    # Ensure the index is within the bounds of the class_names list
    if 0 <= class_idx < len(class_names):
        return class_names[class_idx]
    else:
        # Handle the case where the index is out of bounds
        return "Unknown Class"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image_url = request.form['image_url']
        # Fetch the image from the URL or handle uploaded file
        try:
            image_array = preprocess_image(image_url)
            class_idx = predict_class(image_array)
            # Get the class name based on your dataset
            class_name = get_class_name(class_idx)
            return render_template('result.html', image_url=image_url, class_name=class_name)
        except Exception as e:
            return render_template('error.html', error_message=str(e))
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)