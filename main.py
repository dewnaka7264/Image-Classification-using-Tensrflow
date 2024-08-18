import tensorflow as tf;

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Loading and Preprocessing the Data
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

#Normalizing the Data
train_images = train_images / 255.0
test_images = test_images / 255.0


