

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
from scipy import stats
import tensorflow as tf
import numpy as np
import cv2
import os
import pathlib
import matplotlib.pyplot as plt


data_dir='/workspace/storage/basedata/train/'
def get_list(a):
  cur_dir=os.path.join(data_dir,a)
  filename = os.listdir(cur_dir)
  for i in range(len(filename)):
    filename[i]=os.path.join(cur_dir,filename[i])
  return filename

a='alluvial/'
b='black/'
d='desert/'
r='red/'
f_img_arr=[]
X=[]
labels=[]
X = get_list(a)
l=len(X)
for i in range(l):
  labels.append(0)
X=X+get_list(b)
l1=len(X)
l=l1-l
for i in range(l):
  labels.append(1)
l=l1
X=X+get_list(d)
l1=len(X)
l=l1-l
for i in range(l):
  labels.append(2)
l=l1
X=X+get_list(r)
l1=len(X)
l=l1-l
for i in range(l):
  labels.append(3)
l=l1
X_train, X_test, y_train, y_test = train_test_split( 
            X, labels, test_size = 0.1, random_state=42)





def get_img(filename):  
  img_array=[]
  array_for_median=[]
  label=[]
  med_array=[]
  pre_img=cv2.imread(filename[0])
  pre_img=cv2.resize(pre_img,(7731,7871))
  array_for_median.append(pre_img)

  for i in range(1,len(filename)):
    cur_img=cv2.imread(filename[i])
    cur_img=cv2.resize(cur_img,(7731,7871))
    
    if filename[i][-34:-29]==filename[i-1][-34:-29]:
      img_array.append(cv2.absdiff(cur_img,pre_img))
      if filename[i][34]=='a':
        label.append(0)
      elif filename[i][34]=='b':
        label.append(1)
      elif filename[i][34]=='d':
        label.append(2)
      elif filename[i][34]=='r':
        label.append(3)
    else:
      med_array.append(np.median(np.dstack(array_for_median), -1))
      array_for_median.clear()
    
    array_for_median.append(cur_img)
    pre_img=cur_img
    
    
  
  return img_array,med_array,label



X_train.sort()

f_img_arr,med_array,label= get_img(X_train)

print(label)





X_train, X_valid, y_train, y_valid = train_test_split( 
            f_img_arr, label, test_size = 0.5, random_state=42)




train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
valid_dataset = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))

BATCH_SIZE = 2
SHUFFLE_BUFFER_SIZE = 100

train_ds = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
valid_ds = valid_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)



model = tf.keras.models.Sequential([
                                    tf.keras.layers.Conv2D(8,(15,15),activation='relu',input_shape=(7731,7871,3)),
                                    tf.keras.layers.MaxPool2D(6,6),
                                    # tf.keras.layers.Conv2D(16,(15,15),activation='relu'),
                                    # tf.keras.layers.MaxPool2D(6,6),
                                    # #
                                    # tf.keras.layers.Conv2D(32,(15,15),activation='relu'),
                                    # tf.keras.layers.MaxPool2D(6,6),
                                    # #
                                    # tf.keras.layers.Conv2D(64,(15,15),activation='relu'),
                                    # tf.keras.layers.MaxPool2D(6,6),
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
  validation_data=valid_ds,
  # steps_per_epoch=10,
  epochs=3
)
model.summary()


res=[]
for i in range(len(X_test)):
  img=cv2.imread(X_test[i])
  img=cv2.resize(img,(7731,7871))
  arr=[]
  for med in med_array:
    arr.append(cv2.absdiff(med,img))
  temp_res=model.predict(arr)
  print(temp_res)
  temp2=np.argmax(temp_res, axis=1)
  x = stats.mode(temp2)
  res.append(int(x.mode[0]))
k=0
n=len(res)
for i in range(n):
  if res[i]==y_test[i]:
    k=k+1
print(k/n)