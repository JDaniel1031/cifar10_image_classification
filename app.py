from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import io
import base64
app = Flask(__name__)

# Load your trained model
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'models/cifar10_model.h5')

model = tf.keras.models.load_model(model_path)
class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def preprocess_uploaded_image(image):
    # Resize the uploaded image to match your model requirements
    img = image.resize((32, 32))
    
    # Convert the image to a NumPy array
    img_array = np.array(img) / 255.0  # Normalize the pixel values
    
    # Ensure the image has the correct shape (height x width x channels)
    if img_array.shape != (32, 32, 3):
        raise ValueError("Uploaded image has an incorrect shape. It should be 32x32 pixels with 3 channels (RGB).")
    
    # Add batch dimension to match model input shape
    img_array = np.expand_dims(img_array, axis=0)
    
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
        try:
            uploaded_file = request.files['image']
            if uploaded_file.filename != '':
                image = Image.open(io.BytesIO(uploaded_file.read()))
                image_array = preprocess_uploaded_image(image)
                class_idx = predict_class(image_array)
                class_name = get_class_name(class_idx)
                image_base64 = base64.b64encode(image.tobytes()).decode('utf-8')
                return render_template('result.html', class_name=class_name, image_data=image_base64)
            else:
                return render_template('error.html', error_message="No file uploaded.")
        except Exception as e:
            return render_template('error.html', error_message=str(e))
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
