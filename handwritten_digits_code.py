import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import cv2
import sys

from tensorflow import keras
from tensorflow.keras import layers, datasets, models
from tensorflow.keras.models import Sequential

"""## Prepare Dataset"""

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

print("TRAIN IMAGES: ", train_images.shape)
print("TEST IMAGES: ", test_images.shape)

"""## Create Model"""

num_classes = 10
img_height = 28
img_width = 28

model = Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='sigmoid')
])

"""## Compile Model"""

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

"""## Train Model"""

epochs = 10
history = model.fit(
  train_images,
  train_labels,
  epochs = epochs
)

"""## Visualize Training Results"""

acc = history.history['accuracy']
loss=history.history['loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, loss, label='Loss')
plt.legend(loc='lower right')
plt.title('Training Accuracy and Loss')

"""## Test Image"""

image = (train_images[1]).reshape(1,28,28,1)
# Use predict and argmax to get predicted class
model_pred = np.argmax(model.predict(image, verbose=0), axis=-1)
plt.imshow(image.reshape(28,28))
print('Prediction of model: {}'.format(model_pred[0]))

image = (train_images[2]).reshape(1,28,28,1)
# Use predict and argmax to get predicted class
model_pred = np.argmax(model.predict(image, verbose=0), axis=-1)
plt.imshow(image.reshape(28,28))
print('Prediction of model: {}'.format(model_pred[0]))


# In the Load Model section:

MODEL_PATH = "tf-cnn-model.h5"

def predict_digit(image_path):

    # load model
    model = models.load_model(MODEL_PATH)
    print("[INFO] Loaded model from disk.")

    image = cv2.imread(image_path, 0)
    image1 = cv2.resize(image, (28,28))    # For cv2.imshow: dimensions should be 28x28
    image2 = image1.reshape(1,28,28,1)

    cv2.imshow('digit', image1 )
    pred = np.argmax(model.predict(image2), axis=-1)
    return pred[0]

def main(image_path):
    predicted_digit = predict_digit(image_path)
    print('Predicted Digit: {}'.format(predicted_digit))

if __name__ == "__main__":
    try:
        main(image_path = sys.argv[1])
    except:
        print('[ERROR]: Image not found')


"""## Test Multiple Image"""

images = test_images[1:5]
images = images.reshape(images.shape[0], 28, 28)
print ("Test images array shape: {}".format(images.shape))

# Moved prediction inside the loop where test_image is defined
for i, test_image in enumerate(images, start=1):

    org_image = test_image
    test_image = test_image.reshape(1,28,28,1)
    # Now prediction is calculated within the loop
    prediction = np.argmax(model.predict(test_image, verbose=0), axis=-1)

    print ("Predicted digit: {}".format(prediction[0]))
    plt.subplot(220+i)
    plt.axis('off')
    plt.title("Predicted digit: {}".format(prediction[0]))
    plt.imshow(org_image, cmap=plt.get_cmap('gray'))

plt.show()

"""## Save Model"""

model.save("tf-cnn-model.h5")

"""## Load Model"""

loaded_model = models.load_model("tf-cnn-model.h5")
image = (train_images[2]).reshape(1,28,28,1)
#model_pred = np.argmax(loaded_model.predict(image, verbose=0), axis=-1)

#model_pred = loaded_model.predict_classes(image, verbose=0)
plt.imshow(image.reshape(28,28))
print('Prediction of model: {}'.format(model_pred[0]))
