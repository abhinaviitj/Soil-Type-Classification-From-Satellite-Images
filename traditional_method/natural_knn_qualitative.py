import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from skimage.feature import greycomatrix, greycoprops 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split 

img_name=[]
def return_features(folder) :
    features = []
    filename = os.listdir(folder)
    for i in range(0,40,2):
        img = cv2.imread(os.path.join(folder,filename[i]))
        img_name.append(filename[i])
        img = np.array(img, dtype = np.uint8)
        feat=[]
        img = np.transpose(img,(2,0,1))
        for j in range(len(img)):
            
            gCoMat = greycomatrix(img[j], [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels = None)
            contrast = greycoprops(gCoMat, prop='contrast')
            dissimilarity = greycoprops(gCoMat, prop='dissimilarity')
            homogeneity = greycoprops(gCoMat, prop='homogeneity')
            energy = greycoprops(gCoMat, prop='energy')
            # correlation = greycoprops(gCoMat, prop='correlation')
            f = list(contrast)+list(dissimilarity) +list(homogeneity)+list(energy)
            feat.append(f)
        #img = cv2.imread(os.path.join(folder,filename[i+1]),0)
        #img = np.array(img, dtype = np.uint8)
        #gCoMat = greycomatrix(img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels = None)
        #contrast = greycoprops(gCoMat, prop='contrast')
        #dissimilarity = greycoprops(gCoMat, prop='dissimilarity')
        #homogeneity = greycoprops(gCoMat, prop='homogeneity')
        #energy = greycoprops(gCoMat, prop='energy')
        ## correlation = greycoprops(gCoMat, prop='correlation')
        #f = list(contrast)+list(dissimilarity) +list(homogeneity)+list(energy)
        #feat.append(f)
        fet = feat[0]+feat[1]+feat[2]
        fet = np.array(fet)   
        features.append(fet)
        # print(i/2)
        # i=i+1
        # if i == 41:
        #     break
    return features

all_images = []

images = return_features("alluvial")
print("alluvial done")
all_images = all_images + images

images = return_features("black")
print("black done")
all_images = all_images + images

images = return_features("desert")
print("desert done")
all_images = all_images + images

images = return_features("red")
print("red done")
all_images = all_images + images


# print(all_images)

print("all features calculated")
f=open("features.txt","w+")
X = np.reshape(all_images,(len(all_images),-1))

y=[]
for h in range(80):
    y.append(h)
X_train, X_test, y_train, y_test = train_test_split( 
            X, y, test_size = 0.1, random_state=42) 
Y_test = y_test.copy()
knn = KNeighborsClassifier(n_neighbors=7) 
for i in range(len(y_test)):
    y_test[i]/=20
    y_test[i]=int(y_test[i])
for i in range(len(y_train)):
    y_train[i]/=20
    y_train[i]=int(y_train[i])
knn.fit(X_train, y_train) 
y_res= knn.predict(X_test)
print("Name                                               Actual Class     Predicted Class")
for i in range(len(y_test)):
    print(img_name[Y_test[i]],"         ", y_test[i],"            ", y_res[i])