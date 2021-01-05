import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from skimage.feature import greycomatrix, greycoprops 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split 

def return_features(folder) :
    features = []
    i = 1
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        img = np.array(img, dtype = np.uint8)
        feat=[]
        for j in range(3):
            res = []
            for k in range(len(img)):
                lst = []
                for l in img[k]:
                    lst.append(l[j])
                res.append(lst)
            gCoMat = greycomatrix(res, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels = None)
            contrast = greycoprops(gCoMat, prop='contrast')
            dissimilarity = greycoprops(gCoMat, prop='dissimilarity')
            homogeneity = greycoprops(gCoMat, prop='homogeneity')
            energy = greycoprops(gCoMat, prop='energy')
            # correlation = greycoprops(gCoMat, prop='correlation')
            feat.append( contrast + dissimilarity + homogeneity + energy)
        fet = feat[0]+feat[1] + feat[2]
        print(i)    
        i = i+1
        features= features+ list(fet)
        if i == 21 :
            
            break

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
print("laterite done")
all_images = all_images + images




print("all features calculated")
f=open("features.txt","w+")
#f.write(all_images)
# print()

# all_images = np.float32(all_images)
# print("ready for k means")

# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# flags = cv2.KMEANS_RANDOM_CENTERS
# compactness,labels,centers = cv2.kmeans(all_images, 4, None, criteria, 10, flags)
np.reshape(all_images,(len(all_images),-1))
y=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]

X_train, X_test, y_train, y_test = train_test_split( 
             all_images, y, test_size = 0.1, random_state=42) 


knn = KNeighborsClassifier(n_neighbors=1) 
  
knn.fit(X_train, y_train) 
  
# Predict on dataset which model has not seen before 
print(knn.score(X_test, y_test)) 

# print("labels : ")
# i = 0
# while i < 16 :
#     print(labels[i], labels[i+1], labels[i+2], labels[i +3])
#     i = i + 4