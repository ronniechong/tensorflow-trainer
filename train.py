from dotenv import load_dotenv
load_dotenv()

from flask import Flask, flash, request, redirect, url_for
from flask_ngrok import run_with_ngrok
from flask_cors import CORS
from werkzeug.utils import secure_filename

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import vgg16
from tensorflow.keras import layers, models, Model, optimizers
from tensorflow.keras.preprocessing import image

import numpy as np
import os
import base64

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.secret_key = os.getenv('SECRETKEY')
CORS(app)
# run_with_ngrok(app)
# https://github.com/gstaff/flask-ngrok/issues/2
category_names = os.getenv('CATEGORIES').split(',')
nb_categories = len(category_names)

type = os.getenv('MODE')

if type == 'checkpoint':
  # Load via checkpoints
  img_height, img_width = 200,200
  conv_base = vgg16.VGG16(weights='imagenet', include_top=False, pooling='max', input_shape = (img_width, img_height, 3))
  layers = [
    conv_base,
    layers.Dense(nb_categories, activation='softmax')
  ]
  model = models.Sequential(layers)
  model.load_weights('./model/cp2-0010.ckpt')
else:
  # Load saved model
  model = models.load_model('./model/model_vgg16.h5')

def allowed_file(filename):
  return '.' in filename and \
    filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
  return 'Nothing to see here'

@app.route('/v2/predict', methods=['POST'])
def predictFileUpload():
  if request.method == 'POST':
    print(request)
    if 'file' not in request.files:
      return {
        'Error': 'No file part'
      }
    file = request.files['file']
    if file.filename == '':
      return {
        'Error': 'No selected file'
      }
    if file and allowed_file(file.filename):
      filename = secure_filename(file.filename)
      file.save(os.path.join('./uploads', filename))
      
      img_width, img_height = 200, 200
      img = image.load_img(os.path.join('./uploads', filename), target_size = (img_width, img_height))
      img = image.img_to_array(img)
      img = np.expand_dims(img, axis = 0)

      class_prob=model.predict(img)
      y_pred = np.argmax(class_prob, axis=1)
      count = 0;
      for a in class_prob[0]:
        # print(category_names[count] + ': ' + "{:.2f}".format(a))
        count = count + 1
      return {
        'filename': filename,
        'prediction': category_names[y_pred[0]]
      }
  return 'nothing to see here'

@app.route('/v1/predict', methods=['POST'])
def predictBase64():
  if request.method == 'POST':
    data = request.get_json()
    if data is None:
      return {
        'Error': 'No image'
      }
    else:
      img_data = data['image']
      filename = data['name']
    
    with open(os.path.join('./uploads', filename), "wb") as fh:
      fh.write(base64.decodebytes(img_data.encode()))
      # fh.close()
      
    img_width, img_height = 200, 200
    img = image.load_img(os.path.join('./uploads', filename), target_size = (img_width, img_height))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)

    class_prob=model.predict(img)
    y_pred = np.argmax(class_prob, axis=1)
    count = 0;
    for a in class_prob[0]:
      # print(category_names[count] + ': ' + "{:.2f}".format(a))
      count = count + 1
    return {
      'filename': filename,
      'prediction': category_names[y_pred[0]]
    }
  return 'nothing to see here'

if __name__ == '__main__':
  app.run(host='0.0.0.0')