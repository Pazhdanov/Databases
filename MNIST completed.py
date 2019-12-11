from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2)

from mlxtend.data import loadlocal_mnist
x_train, y_temp = loadlocal_mnist(
images_path="C://Users//pazh2//Documents//Python Scripts//train-images-idx3-ubyte" ,
labels_path="C://Users//pazh2//Documents//Python Scripts//train-labels-idx1-ubyte")

x_test, y_temp2 = loadlocal_mnist(
images_path="C://Users//pazh2//Documents//Python Scripts//t10k-images-idx3-ubyte" ,
labels_path="C://Users//pazh2//Documents//Python Scripts//t10k-labels-idx1-ubyte")

import numpy as np

x_test = x_test / 255
x_train = x_train / 255

y_train = np.zeros((60000, 10))
for i in range (60000):
    y_train[i][y_temp[i]] = 1.0
        
y_test = np.zeros((10000, 10))
for i in range (10000):
    y_test[i][y_temp2[i]] = 1.0
        
model = Sequential()
model.add(Dense(40, input_dim = 784, activation = "relu"))
model.add(Dense(160, activation = "relu"))
model.add(Dense(10, activation = "sigmoid"))

model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])

model.fit(x_train, y_train, epochs = 70, batch_size = 1000)

scores = model.evaluate(x_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#n = 2601
#Xtat = x_test.reshape(10000, 784)
#plt.imshow(Xtat[n].reshape(28,28),cmap="gray")
#print(y_temp2[n])