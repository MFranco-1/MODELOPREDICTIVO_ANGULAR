# =========================
# IMPORTACIÓN DE LIBRERÍAS
# =========================
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# =========================
# CONFIGURACIÓN DE APP
# =========================
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# =========================
# CARGA DEL MODELO TFLITE
# =========================
interpreter = tf.lite.Interpreter(model_path='brain_tumor_cnn.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']


# =========================
# FUNCIÓN DE PREDICCIÓN
# =========================
def predict_with_tflite(img_array):
    img_array = img_array.astype(np.float32) / 255.0

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0]


# =========================
# ENDPOINT DE CLASIFICACIÓN
# =========================
@app.route('/api/clasificar', methods=['POST'])
def clasificar_api():

    # Validación de archivo
    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'No se envió imagen'}), 400

    # Guardar imagen
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Preprocesamiento
    img = image.load_img(filepath, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Predicción
    prediction = predict_with_tflite(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    probabilities = {
        class_names[i]: float(f"{prob:.4f}")
        for i, prob in enumerate(prediction)
    }

    # =========================
    # GENERAR GRÁFICA
    # =========================
    plt.figure(figsize=(6, 4))
    plt.bar(probabilities.keys(), probabilities.values())
    plt.title('Probabilidades por clase')
    plt.ylabel('Confianza')
    plt.tight_layout()

    graph_path = os.path.join(app.config['UPLOAD_FOLDER'], 'probabilidades.png')
    plt.savefig(graph_path)
    plt.close()

    # =========================
    # RESPUESTA
    # =========================
    return jsonify({
        'prediction': f'Predicción: {predicted_class.upper()}',
        'image_name': filename,
        'probs': probabilities
    })


# =========================
# EJECUCIÓN DEL SERVIDOR
# =========================
if __name__ == '__main__':
    app.run(debug=True, port=5000)