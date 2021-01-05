print('start')
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

# Vegetation removal as image preprocessing base file


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
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size = 0.1, random_state=42)



print('train-test-split')
print(len(X_test))




def get_img(filename):  
  
  img_array=[]
  for i in range(0,len(filename)):
    img=cv2.imread(filename[i])
    img=cv2.resize(img,(1000,1000))
    # cur_img=cur_img.astype(np.float32)
    
    height, width, depth = img.shape
    mask = np.zeros((height, width), dtype = img.dtype)

    for i in range(height) :
      for j in range(width) :
        if(   img[i][j][0] > 100
          and img[i][j][0] < 200
          and img[i][j][1] > 100
          and img[i][j][1] < 200
          and img[i][j][2] > 100
          and img[i][j][2] < 200) :
            mask[i][j] = 1


    temp_img = cv2.bitwise_and(img, img, mask=mask)
    temp_img=temp_img.astype(np.float32)
    temp_img=temp_img/255
    if type(img_array)==list:
      img_array=np.array([temp_img])
    else:
      img_array=np.append(img_array,[temp_img],axis=0)
  
  return img_array




X_train= get_img(X_train)



print(X_train.shape)

print('absdiff calculated')






onehot=[]
for value in y_train:
	letter = [0 for _ in range(4)]
	letter[value] = 1
	onehot.append(np.array(letter))

y_train=np.array(onehot)


print('one-hot endoded')


model = tf.keras.models.Sequential([
                                      # tf.keras.layers.experimental.preprocessing.Rescaling(1./255),

                                    tf.keras.layers.Conv2D(8,(15,15),activation='relu',input_shape=(1000,1000,3)),
                                    tf.keras.layers.MaxPool2D(4,4),
                                    tf.keras.layers.Conv2D(16,(15,15),activation='relu'),
                                    tf.keras.layers.MaxPool2D(4,4),
                                    # #
                                    tf.keras.layers.Conv2D(32,(15,15),activation='relu'),
                                    tf.keras.layers.MaxPool2D(4,4),
                                    # #
                                    # tf.keras.layers.Conv2D(64,(15,15),activation='relu'),
                                    # tf.keras.layers.MaxPool2D(4,4),
                                    # tf.keras.layers.Conv2D(128,(15,15),activation='relu'),
                                    # tf.keras.layers.MaxPool2D(4,4),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(256,activation='relu'),
                                    tf.keras.layers.Dense(4,activation='softmax')
])
# model.summary()

model.compile(loss='categorical_crossentropy',
optimizer=RMSprop(lr=0.001),
metrics=['accuracy'])



model.summary()

model.fit(
  x=X_train,
  y=y_train,
  validation_split=0.11,
  # validation_data=val_ds,
  steps_per_epoch=10,
  epochs=3
)

print('model fitted')




print('testing starts')

# model.predict(test_ds)
res=[]
for i in range(len(X_test)):
  img=cv2.imread(X_test[i])
  img=cv2.resize(img,(1000,1000))
  arr=[]
  height, width, depth = img.shape
  mask = np.zeros((height, width), dtype = img.dtype)

  for i in range(height) :
    for j in range(width) :
      if(   img[i][j][0] > 100
        and img[i][j][0] < 200
        and img[i][j][1] > 100
        and img[i][j][1] < 200
        and img[i][j][2] > 100
        and img[i][j][2] < 200) :
          mask[i][j] = 1


  arr = cv2.bitwise_and(img, img, mask=mask)
  temp_res=model.predict(np.array([arr]))
  print(temp_res)
  temp2 = np.argmax(temp_res, axis=1)
  res.append(int(temp2))

k=0
n=len(res)
for i in range(n):
  if res[i]==y_test[i]:
    k=k+1
print('accuracy=',k/n)
print(3/0)
