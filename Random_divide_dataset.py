
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf
import numpy as np
import cv2
import os
import pathlib
import matplotlib.pyplot as plt

#program to split dataset randomly to test, train and validation datasets.

data_dir='/workspace/storage/basedata/test/'
data_dir = pathlib.Path(data_dir)
AUTOTUNE = tf.data.experimental.AUTOTUNE
image_count=280
list_ds = tf.data.Dataset.list_files(str('/workspace/storage/basedata/all/*/*'), shuffle=False)
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]))
print(class_names)

val_size = int(image_count * 0.23)
tn_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)

train_count=tf.data.experimental.cardinality(tn_ds).numpy()
test_size = int(train_count * 0.25)
train_ds = tn_ds.skip(test_size)
test_ds = tn_ds.take(test_size)

print(tf.data.experimental.cardinality(train_ds).numpy())
print(tf.data.experimental.cardinality(val_ds).numpy())
print(tf.data.experimental.cardinality(test_ds).numpy())

batch_size = 1
img_height = 7781
img_width = 7871

def get_label(file_path):
  # convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  one_hot = parts[-2] == class_names
  # Integer encode the label
  return tf.argmax(one_hot)

def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # resize the image to the desired size
  return tf.image.resize(img, [img_height, img_width])

def process_path(file_path):
  label = get_label(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
train_ds = train_ds.map(process_path)
val_ds = val_ds.map(process_path)
test_ds = test_ds.map(process_path)

for image, label in train_ds.take(1):
  print("Image shape: ", image.numpy().shape)
  print("Label: ", label.numpy())

def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)
test_ds = configure_for_performance(test_ds)



model = tf.keras.models.Sequential([tf.keras.layers.experimental.preprocessing.Rescaling(1./255),

                                    tf.keras.layers.Conv2D(8,(15.15),activation='relu'),
                                    tf.keras.layers.MaxPool2D(6,6),
                                    tf.keras.layers.Conv2D(16,(15,15),activation='relu'),
                                    tf.keras.layers.MaxPool2D(6,6),
                                    #
                                    tf.keras.layers.Conv2D(32,(15,15),activation='relu'),
                                    tf.keras.layers.MaxPool2D(6,6),
                                    #
                                    tf.keras.layers.Conv2D(64,(15,15),activation='relu'),
                                    tf.keras.layers.MaxPool2D(6,6),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(256,activation='relu'),
                                    tf.keras.layers.Dense(1,activation='softmax')
])
# model.summary()

model.compile(loss='categorical_crossentropy',
optimizer=RMSprop(lr=0.001),
metrics=['accuracy'])

model.fit(
  train_ds,
  validation_data=val_ds,
  steps_per_epoch=10,
  epochs=3
)

model.predict(test_ds)