import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
# TODO: Make all other necessary imports.

import numpy as np
import matplotlib.pyplot as plt
import json
import tensorflow.keras.models
from PIL import Image
import argparse

#python predict.py ./test_images/wild_pansy.jpg my_model.h5 --top_k 3 --category_names_map label_map.json

my_parser = argparse.ArgumentParser(description='Sample image classifier application')



my_parser.add_argument('image_path', help='Enter the path of the image for which the prediction is required')

my_parser.add_argument('model', help='Enter the path of the model that you want to use')

my_parser.add_argument('--category_names_map', default = 'label_map.json', help='Enter the labelling map')

my_parser.add_argument('--top_k', default= '3', help='Enter the number of classes for which probabilities will be displayed', type=int)

commands = my_parser.parse_args()

print('If you read this line it means that you have provided all the parameters')


#Get the arguments into the variables

image_path, model, K, category_names_map = commands.image_path, commands.model, commands.top_k, commands.category_names_map


#Process the image


def process_image(img):
    image_size = 224
    tf_img = np.squeeze(img)
    #This helps in normalizing the image
    normalized_img = tf_img / 255
    
    resized_image = tf.image.resize(normalized_img, (image_size, image_size))
    
    #We cannot use reshape here since we actually have to reduce the size. Reshape keeps the number of elements same
    #resized_image = tf.reshape(normalized_img, (224,224))
    
    return resized_image.numpy()

#Get the trained model
def get_trained_model(model):
	trained_model = tf.keras.models.load_model(model, custom_objects={'KerasLayer': hub.KerasLayer} ,compile=False)
	return trained_model

#Get class names

def get_class_names(map):
    with open(map, 'r') as f:
        class_names = json.load(f)
    return class_names

#The predict function

def predict(image_path, trained_model, top_k):
    
    #Load the image into the variable
    im = Image.open(image_path)
    test_image = np.asarray(im)
    
    #Process the image using the function
    processed_test_image = process_image(test_image)
    
    #Get the probabilities from the trained model
    probabilities = trained_model.predict(np.expand_dims(processed_test_image, axis=0))
    
    #Get the top 5 probabilities
    top_values, top_indices = tf.math.top_k(probabilities, top_k)
    
    #Get the classes names
    classes = [class_names[str(value+1)] for value in top_indices.cpu().numpy()[0]]

    #return top_values.numpy()[0], classes
    return classes



if __name__ == "__main__":
    
    trained_model = get_trained_model(model)
    class_names = get_class_names(category_names_map)
    
    print(predict(image_path, trained_model, K))
    
    
    
    