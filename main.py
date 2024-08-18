import tensorflow as tf;

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Loading and Preprocessing the Data
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

#Normalizing the Data
train_images = train_images / 255.0
test_images = test_images / 255.0

#Convolutional nural network

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

#compilation of the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


#training of the model
history = model.fit(train_images, train_labels, epochs=5,
                    validation_data=(test_images, test_labels))


#evaullation of the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")

#plotting the training results
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.8, 1])
plt.legend(loc='lower right')
plt.show()





