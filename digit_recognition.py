"'In this model we have taken two hidden two layers and 1 i/p layer, 1 o/p layer and here we train 28 x 28 pixels images'"
import cv2 as cv           #To get my example loaded in the program
import numpy as np        #This is used for matrix
import tensorflow as tf     #This is used to type the code
import matplotlib.pyplot as plt  #For non linear functions



# mnist = tf.keras.datasets.mnist #This is used to get the training data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data() #It splits the data into training data & test data
x_train = tf.keras.utils.normalize(x_train, axis=1)     #This is to normalize the o/p i.e to bring total output to 1
x_test = tf.keras.utils.normalize(x_test, axis=1)
model = tf.keras.models.Sequential()                  #Sequential model is loaded
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))    #i/p layer
model.add(tf.keras.layers.Dense(units = 128, activation = tf.nn.relu))  #hidden layer1
model.add(tf.keras.layers.Dense(units = 128, activation = tf.nn.relu))  #hidden layer2
model.add(tf.keras.layers.Dense(units=10, activation = tf.nn.softmax))  #o/p layer
model.compile(optimizer= 'adam', loss= 'sparse_categorical_crossentropy', metrics = ['accuracy'])   #here my loss and accuracy is calcutlated
model.fit(x_train, y_train, epochs = 10)                                                             #here we tell to train the network 3 times on the same data
loss, accuracy = model.evaluate(x_test, y_test)                                                     #here our model is trainded i.e accuracy and loss is calculated
print(accuracy*100)
print(loss*100)
model.save('digits.model')                  #here we store the data of our neural network

for x in range(1,6):
    img = cv.imread(f'{x}.png')[:,:,0]
    img =np.invert( np.array([img]))
    img = tf.keras.utils.normalize(img, axis=1)
    prediction = model.predict(img)
    print(np.argmax(prediction))
    # img = tf.keras.utils.normalize(img , axis = 1)
    # result = model.predict(img)
    # print(np.argmax(result))
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()







