# Tensorflow module, Helper libraries import
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Fashion MNIST Dataset Import_Train 60,000 images, Test 10,000 images
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirts/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Data research
print(train_images.shape)
print(len(train_labels))
print(train_labels)
print(test_images.shape)
print(len(test_labels))

# Data Pretreatment_Train set's first image pixel range is 0~255
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
#  Adjust the range of these values between 0 and 1 before injecting them into the neural network. 
train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

### Model Configuration
#   layer setting
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
    ])






#   model compile
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])



### Model Training
#   model fit
model.fit(train_images, train_labels, epochs=10)


#   Accuracy Evaluation
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("\nTest accuracy :", test_acc)


#   Predict
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)


print(predictions[0])
print('\n')


print(np.argmax(predictions[0]))
print('\n')

print(test_labels[0])
print('\n')
