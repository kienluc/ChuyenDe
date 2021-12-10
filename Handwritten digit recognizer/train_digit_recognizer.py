import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization,Activation,Input
from keras.layers import Conv2D, MaxPooling2D, Lambda
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.regularizers import l2
from keras.utils import np_utils
import sklearn.metrics as metrics

import itertools


# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

input_shape = (28, 28, 1)


# Normalise
x_train = x_train.astype('float32')
x_train /= 255
x_test = x_test.astype('float32')
x_test /= 255

num_classes = len(np.unique(y_train))

# One hot encoding
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)


batch_size = 128
epochs = 20


model = Sequential([
Conv2D(filters = 32, kernel_size = 5, strides = 1, activation = 'tanh', input_shape = (28,28,1), kernel_regularizer=l2(0.0005), name = 'convolution_1'),
Conv2D(filters = 32, kernel_size = 5, strides = 1, name = 'convolution_2', use_bias=False),
BatchNormalization(name = 'batchnorm_1'),
    
# -------------------------------- #  
Activation("relu"),
MaxPooling2D(pool_size = 2, strides = 2, name = 'max_pool_1'),
Dropout(0.25, name = 'dropout_1'),
# -------------------------------- #  
    
# Layer 3
Conv2D(filters = 64, kernel_size = 3, strides = 1, activation = 'tanh', kernel_regularizer=l2(0.0005), name = 'convolution_3'),
    
# Layer 4
Conv2D(filters = 64, kernel_size = 3, strides = 1, name = 'convolution_4', use_bias=False),
    
# Layer 5
BatchNormalization(name = 'batchnorm_2'),
    
# -------------------------------- #  
Activation("tanh"),
MaxPooling2D(pool_size = 2, strides = 2, name = 'max_pool_2'),
Dropout(0.25, name = 'dropout_2'),
Flatten(name = 'flatten'),
# -------------------------------- #  
    
# Layer 6
Dense(units = 256, name = 'fully_connected_1', use_bias=False),
    
# Layer 7
BatchNormalization(name = 'batchnorm_3'),

# -------------------------------- #  
Activation("tanh"),
# -------------------------------- #  
    
# Layer 8
Dense(units = 128, name = 'fully_connected_2', use_bias=False),
    
# Layer 9
BatchNormalization(name = 'batchnorm_4'),
    
# -------------------------------- #  
Activation("tanh"),
# -------------------------------- #  
    
# Layer 10
Dense(units = 84, name = 'fully_connected_3', use_bias=False),
    
# Layer 11
BatchNormalization(name = 'batchnorm_5'),
    
# -------------------------------- #  
Activation("tanh"),
Dropout(0.25, name = 'dropout_3'),
# -------------------------------- #  

# Output
Dense(units = num_classes, activation = 'softmax', name = 'output')
    
])

model._name = 'LeNet5v2'

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs= epochs, batch_size= batch_size, verbose=1, \
                    validation_data=(x_test, y_test))
print("The model has successfully trained")

def plotgraph(epochs, acc, val_acc):
    # Plot training & validation accuracy values
    plt.plot(epochs, acc, 'b')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(acc)+1)

plotgraph(epochs, acc, val_acc)
plotgraph(epochs, loss, val_loss)


y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5)

cm = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


class_names = [i for i in range(num_classes)]
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(figsize=(50,20))
plot_confusion_matrix(cm, classes=class_names,
                      title='Confusion matrix, without normalization')
plt.show()


model.save('mnist.h5')
print("Saving the model as mnist.h5")

