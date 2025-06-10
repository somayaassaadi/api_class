from flask import Flask, request, render_template
#Fournit le modèle MobileNetV2 et les outils pour le traitement d'image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
#Manipuler les images (ouvrir, convertir, redimensionner, etc.)
from PIL import Image
import os

app = Flask(__name__)
model = MobileNetV2(weights='imagenet')
img = Image.open("./pho.jpeg")

def model_predict(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    X = preprocess_input(x, data_format=None) 
    preds = model.predict(x)
    return decode_predictions(preds, top=1)[0][0]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join('static', file.filename)
            file.save(filepath)
            pred_class = model_predict(filepath)
            label = pred_class[1].replace('_', ' ').capitalize()
            prob = f"{pred_class[2]*100:.2f}%"
            prediction = f"{label} ({prob})"
            return render_template('index.html', prediction=prediction, image_path=filepath)
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run("0.0.0.0", port=8081, debug=True)
