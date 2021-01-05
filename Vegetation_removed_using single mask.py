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

# Vegetation removal as image preprocessing


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

# vegetation removal on one image of block and using them for rest all images

no_of_loops=5
acc=0
for t in range(no_of_loops):
	X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size = 0.2, random_state=42)



	print('train-test-split')
	print(len(X_test))




	def get_img(filename):
		label=[0]
		pre_img=cv2.imread(filename[0])
		img=cv2.resize(pre_img,(1000,1000))

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
		
		
		img_array=[]
		img_array=np.array([temp_img])
		for i in range(1,len(filename)):
			cur_img=cv2.imread(filename[i])
			img=cv2.resize(cur_img,(1000,1000))
			if filename[i][-34:-29]!=filename[i-1][-34:-29]:
				height, width, depth = img.shape
				mask = np.zeros((height, width), dtype = img.dtype)
				for k in range(height) :
					for j in range(width) :
						if(   img[k][j][0] > 100
							and img[k][j][0] < 200
							and img[k][j][1] > 100
							and img[k][j][1] < 200
							and img[k][j][2] > 100
							and img[k][j][2] < 200) :
								mask[k][j] = 1

			temp_img = cv2.bitwise_and(img, img, mask=mask)
			temp_img=temp_img.astype(np.float32)
			temp_img=temp_img/255
			img_array=np.append(img_array,[temp_img],axis=0)

			if filename[i][34]=='a':
				label.append(0)
			elif filename[i][34]=='b':
				label.append(1)
			elif filename[i][34]=='d':
				label.append(2)
			elif filename[i][34]=='r':
				label.append(3)    
		return img_array,label



	X_train.sort()

	X_train,y_train= get_img(X_train)






	onehot=[]
	for value in y_train:
		letter = [0 for _ in range(4)]
		letter[value] = 1
		onehot.append(np.array(letter))
	
	y_train=np.array(onehot)


	print('one-hot endoded')
	X_train, X_valid, y_train, y_valid = train_test_split( 
            X_train, y_train, test_size = 0.125, random_state=42)

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
		# validation_split=0.125,
		validation_data=(X_valid, y_valid),
		steps_per_epoch=10,
		epochs=10,
	)

	print('model fitted')

	# model_fit=model.fit(x=train_ds,
	#                     steps_per_epoch=10,
	#                     epochs=10,
	#                     validation_steps=10,
	#                     validation_data=val_ds
	#                   )


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
		temp2 = np.argmax(temp_res, axis=1)
		res.append(int(temp2))
	print(res)
	k=0
	n=len(res)
	for i in range(n):
		if res[i]==y_test[i]:
			k=k+1
	k=k/n
	print('accuracy=',k)
	acc=acc+k
acc=acc/no_of_loops
print('final accuracy=', acc)