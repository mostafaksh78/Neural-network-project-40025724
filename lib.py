import numpy as np
import tensorflow as tf
import pandas as pd
import math
from keras.metrics import TruePositives
import keras.backend as K
from keras import models,layers
from keras.layers import Layer
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
plt.style.use('ggplot')
import time
# from rbf_keras import RBFLayer,InitCentersRandom
epochs = 50
name = "Result_RBF"
class RBFLayer(Layer):
    def __init__(self, units, gamma, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.gamma = K.cast_to_floatx(gamma)
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'units': self.units,
            'gamma': self.gamma,
        })
        return config
    def build(self, input_shape):
#         print(input_shape)
#         print(self.units)
        self.mu = self.add_weight(name='mu',
                                  shape=(int(input_shape[1]), self.units),
                                  initializer='uniform',
                                  trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        diff = K.expand_dims(inputs) - self.mu
        l2 = K.sum(K.pow(diff, 2), axis=1)
        res = K.exp(-1 * self.gamma * l2)
        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

data = pd.read_csv("data.csv")
data = np.array(data,dtype=float)
labels = data[:,-1]
labels = np.array(labels)
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels)
data =data[:,:-2]
for i in range(np.shape(data)[1]):
    i = 0
    a = data[:,i]
    a = a - np.mean(a)
    m = np.max(a)
    mi = np.min(a)
    b = (a - mi)/(m-mi)
    data[:,i] = b
xTrain,xTest,yTrain,yTest = train_test_split(data,labels,test_size=0.2)
print(np.shape(xTrain))
net = models.Sequential([
    layers.Input(shape = (9,)),
    RBFLayer(units=10,gamma=0.5),
    # layers.Dense(15,activation= "sigmoid"),
    layers.Dense(2,activation="softmax")
])
net.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),loss=tf.keras.losses.CategoricalCrossentropy(),metrics=['acc',f1_m,precision_m, recall_m])
net.summary()

history = net.fit(xTrain,yTrain,batch_size=1,epochs=epochs,validation_data=(xTest,yTest))
train_acc = history.history['acc']
val_acc = history.history['val_acc']
train_loss = history.history['loss']
val_loss = history.history['val_loss']
val_f1 = history.history['val_f1_m']
train_f1 = history.history['f1_m']
train_precision = history.history['precision_m']
train_recall = history.history['recall_m']
val_precision = history.history['val_precision_m']
val_recall = history.history['val_recall_m']

epochs = range(1,epochs + 1)
fig, (ax1, ax2,ax3,ax4,ax5) = plt.subplots(5,1)
# fig.suptitle('Horizontally stacked subplots')
ax1.plot(epochs, train_acc, 'g', label='Training Accuracy')
ax1.plot(epochs, val_acc, 'b', label='validation Accuracy')
ax2.plot(epochs, train_loss, 'g', label='Training loss')
ax2.plot(epochs, val_loss, 'b', label='validation loss')
ax1.set_title('Training and Validation Accuracy')
ax2.set_title('Training and Validation Loss')
ax1.set_xlabel('Epochs')
ax2.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax2.set_ylabel('Loss')
ax3.plot(epochs, train_f1,'g',label = "Train F1 score")
ax3.plot(epochs, val_f1,'b',label = "Validation F1 score")
ax3.set_title('Training and Validation F1 score')
ax3.set_xlabel(epochs)
ax3.set_ylabel('F1')
ax4.plot(epochs, train_precision,'g',label = "Train  Preccision")
ax4.plot(epochs, val_precision,'b',label = "Validation  Preccision")
ax4.set_title('Training and Validation  Preccision')
ax4.set_xlabel(epochs)
ax4.set_ylabel('Preccision')
ax5.plot(epochs, train_recall,'g',label = "Train  ")
ax5.plot(epochs, val_recall,'b',label = "Validation  ")
ax5.set_title('Training and Validation  Recall')
ax5.set_xlabel(epochs)
ax5.set_ylabel('Recall')
# plt.title("")
plt.legend()
net.save(name+".h5")
plt.savefig(name + ".png")
plt.show()
plt.close(fig)




