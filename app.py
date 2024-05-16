import os
import socket
from typing import List
import app
import numpy as np
import tensorflow
from flask import Flask, jsonify
from keras import Sequential
from numpy.linalg import norm
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.preprocessing import image

localhost_ip = socket.gethostbyname('localhost')
print("Localhost IP address:", localhost_ip)
app = Flask(__name__)


@app.route ( '/file:///D:/Final%20%20Project/frontend/index.html', methods=["GET"] )
def get_data():
    data = {"key": "value"}
    return data


def index():
    return "<h1>Fashion Recommender System</h1>"


def get_recommendations():
    recommendations = ["item1", "item2", "item3"]
    return jsonify ( recommendations )


if __name__ == '__main__':
    app.run(debug=True)
    app.run (debug=True, host='127.0.0.1', port=8000)

model = ResNet50 ( weights='imagenet', include_top=False, input_shape=(224, 224, 3) )
model.trainable = False

model: Sequential = tensorflow.keras.Sequential ( [
    model,
    GlobalMaxPooling2D ()] )

print ( model.summary () )


def extract_features(img_path: object, models: object) -> object:
    """

    :type models: object
    """
    img = image.load_img ( img_path, target_size=(224, 224) )
    img_array = image.img_to_array ( img )
    expanded_img_array = np.expand_dims ( img_array, axis=0 )
    preprocessed_img = preprocess_input ( expanded_img_array )
    result = models.predict ( preprocessed_img ).flatten ()
    normalized_result = result / norm ( result )

    return normalized_result


filenames: List[str] = []

for file in os.listdir ( 'images' ):
    filenames.append ( os.path.join ( 'images', file ) )

feature_list = []
