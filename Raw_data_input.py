from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf
import numpy as np
import cv2
import os

os.environ['CUDA_VISIBLE_DEVICES']='3'
print(tf.__version__)

train = ImageDataGenerator(rescale = 1/255)
validation = ImageDataGenerator(rescale = 1/255)

train_dataset = train.flow_from_directory('basedata/train/',
                                         target_size=(5000,5000), 
                                         batch_size=3,
                                         class_mode = 'categorical')
valid_dataset = validation.flow_from_directory('basedata/validation/',
                                         target_size=(5000,5000),
                                         batch_size=3,
                                         class_mode = 'categorical')

model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(8,(15,15),activation='relu', input_shape=(5000,5000,3)),
                                    tf.keras.layers.MaxPool2D(8,8),
                                    tf.keras.layers.Conv2D(16,(15,15),activation='relu'),
                                    tf.keras.layers.MaxPool2D(8,8),
                                    #
                                    tf.keras.layers.Conv2D(32,(15,15),activation='relu'),
                                    tf.keras.layers.MaxPool2D(8,8),
                                    #
                                    #tf.keras.layers.Conv2D(64,(15,15),activation='relu'),
                                    #tf.keras.layers.MaxPool2D(8,8),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(256,activation='relu'),
                                    tf.keras.layers.Dense(1,activation='softmax')
])

model.compile(loss='categorical_crossentropy',
optimizer=RMSprop(lr=0.001),
metrics=['accuracy'])

model_fit=model.fit(x=train_dataset,
                    steps_per_epoch=10,
                    epochs=10,
                    validation_steps=10,
                    validation_data=valid_dataset,
                    batch_size=1)

dir_path = 'basedata/test'
for i in os.listdir(dir_path):
    img=image.load_img(dir_path+'//'+i,target_size=(5000,5000))
    print(i)
    x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    images=np.vstack([x])
    val=model.predict_classes(images)
    print(val)
