from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load your model
model = load_model(r"C:\Users\tanis\OneDrive\Desktop\capstone\cnn_model.h5")

# Define label mapping
labels = {0: "COVID-19", 1: "Normal"}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Save uploaded file
        file = request.files['file']
        if file:
            # Create the uploads directory if it doesn't exist
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Preprocess the image
            img = load_img(filepath, target_size=(224, 224))
            img = img_to_array(img) / 255.0
            img = np.expand_dims(img, axis=0)

            # Make prediction
            prediction = model.predict(img)
            result = labels[np.argmax(prediction)]

            # Update the image path to be relative to the static folder
            image_path = f'uploads/{file.filename}'

            return render_template('index.html', prediction=result, image=image_path)

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
