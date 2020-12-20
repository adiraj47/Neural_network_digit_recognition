#import the libraries
import cv2 as cv
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#loading the data
mnist = tf.keras.datasets.mnist
#creating a neural network
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
print(x_test.shape)
#print(x_train[0]) shows what imgae values it contain
plt.imshow(x_train[0])
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
y_train_one_hot =tf.keras.utils.to_categorical(y_train)
y_test_one_hot = tf.keras.utils.to_categorical(y_test)
#print(y_train_one_hot[0]) to convt the into 0 to 9 representation eg if 2 [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
model= tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(64, kernel_size= 3, activation= 'relu', input_shape=(28,28,1)))
model.add(tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.compile(optimizer= 'adam', loss='categorical_crossentropy', metrics=['accuracy'])
hist = model.fit(x_train, y_train_one_hot, validation_data=(x_test, y_test_one_hot), epochs= 3)
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc= 'upper left')
plt.show()
predictions = model.predict(x_test[:4])
print(np.argmax(predictions, axis=1))
print(y_test[:4])

#To enter my digits
# for x in range(1,6):
#     img = cv.imread(f'{x}.png')[:,:,0]
#
#     img =np.invert( np.array([img]))
#     print(img.shape)
#     prediction = model.predict(img)
#     print(np.argmax(prediction))

