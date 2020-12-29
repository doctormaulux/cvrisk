
from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
print("tensorflow_version: ", tf.__version__)
import glob
import sklearn.model_selection as sk
#import os
import imageio
import os
import pandas as pd

path = os.getcwd()




# DATA LOADING



images=[]
labels=[]
#filesctrl2 = glob.glob (r"C:\Users\giuseppe.maulucci\OneDrive - Università Cattolica del Sacro Cuore\Projects\cvrisk\ImageSequenceGP\ImageSequence_Ctrl2\results\*.tif")
#filesctrl = glob.glob (r"C:\Users\giuseppe.maulucci\OneDrive - Università Cattolica del Sacro Cuore\Projects\cvrisk\ImageSequenceGP\ImageSequence_Ctrl\results\*.tif")
#filesIGT = glob.glob(r"C:\Users\giuseppe.maulucci\OneDrive - Università Cattolica del Sacro Cuore\Projects\cvrisk\ImageSequenceGP\ImageSequence_IGT\results\*.tif") 
filesDM2 = glob.glob(path+"\\ImageSequenceGP\\ImageSequence_DM2\\results\\*.tif") 
filesMCV = glob.glob(path+"\\ImageSequenceGP\\ImageSequenceMCV\\results\\*.tif") 

direct =[filesDM2, filesMCV] 


class_names=["DM2","MCV"]
nclass=2



for i in range(nclass):
    for filename in direct[i]: 
        im = imageio.imread(filename)
        images.append(im)
        labels.append(i)

print("#images:",len(images))
print("#DM2:",len(filesDM2))
print("#MCV:",len(filesMCV))
#print("#CTRL:",len(filesctrl))

#SHUFFLING AND TEST-TRAIN SPLITTING
im_train, im_test, label_train, label_test = sk.train_test_split(images,labels,test_size=0.30, random_state=47)

"""
#OVERSAMPLING 

#combine them back for resampling
train_data = pd.concat([im_train, label_train], axis=1)# separate minority and majority classes
negative = train_data[train_data.diagnosis==0]
positive = train_data[train_data.diagnosis==1]# upsample minority
pos_upsampled = resample(positive,
 replace=True, # sample with replacement
 n_samples=len(negative), # match number in majority class
 random_state=27) # reproducible results# combine majority and upsampled minority
upsampled = pd.concat([negative, pos_upsampled])# check new class counts

"""



#NORMALIZATION
train_images=np.array(im_train)
train_labels=np.array(label_train)
test_images=np.array(im_test)
test_labels=np.array(label_test)
train_images=train_images/4096
test_images=test_images/4096
#train_images= np.expand_dims(train_images, axis=-1)
#test_images= np.expand_dims(test_images, axis=-1)








#VISUALIZZAZIONE DATI TRAIN E SUMMARY DATI PER MODELLO
plt.figure(figsize=(15,15))
for i in range(100):
    plt.subplot(10,10,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    
    plt.xlabel(class_names[train_labels[i]],fontsize=10)
plt.show()

#imageio.imsave("xxx.png",train_images[3])

len(train_labels)
len(test_labels)
test_labels
print('train images_data shape:', np.array(train_images).shape)
print('test images_data shape:', np.array(test_images).shape)


#DEFINIZIONE DEL MODELLO
"""
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(110,110,3)),
    keras.layers.Dense(20,activation=tf.nn.relu),
  #  keras.layers.Dense(100, activation=tf.nn.relu),
 #   keras.layers.Dense(100, activation=tf.nn.relu),
    keras.layers.Dense(nclass,activation=tf.nn.softmax)
])

"""
model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(110, 110, 3)))
model.add(layers.MaxPooling2D((3, 3)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Conv2D(32, (2, 2), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(nclass, activation="softmax"))

model.summary()


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))


#TEST PERFORMANCE MODELLO
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('Test accuracy:', test_acc)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.4, 1])
plt.legend(loc='lower right')


predictions = model.predict(test_images)
#tr_images= np.squeeze(train_images)
#te_images= np.squeeze(test_images)
#te_labels= test_labels.tolist()


maxima=[]
risk=[]
for i in predictions:
    maxi=np.argmax(i)
    prob=i[0]
    maxima.append(maxi)
    risk.append(prob)
    
risk

tf.math.confusion_matrix(
    test_labels, maxima, num_classes=None, weights=None, dtype=tf.dtypes.int32,
    name=None)
