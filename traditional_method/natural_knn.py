import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from skimage.feature import greycomatrix, greycoprops 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split 

def return_features(folder) :
    features = []
    filename = os.listdir(folder)
    for i in range(0,40,2):
        img = cv2.imread(os.path.join(folder,filename[i]))
        
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
            print(contrast.shape,dissimilarity.shape,homogeneity.shape,energy.shape)
            print(type(contrast))
            f = list(contrast)+list(dissimilarity) +list(homogeneity)+list(energy)
            feat.append(f)
        fet = feat[0]+feat[1]+feat[2]
        fet = np.array(fet)   
        features.append(fet)
        
    return features

all_images = []

images = return_features("alluvial1")
print("alluvial done")
all_images = all_images + images

images = return_features("black1")
print("black done")
all_images = all_images + images

images = return_features("desert1")
print("desert done")
all_images = all_images + images

images = return_features("red1")
print("red done")
all_images = all_images + images


# print(all_images)

print("all features calculated")
f=open("features.txt","w+")
X = np.reshape(all_images,(len(all_images),-1))

y=[]
for h in range(4):
    for k in range(20):
        y.append(h)
print(y)
m=0
res=[]
for j in range(3,9,2):
    m=0
    for i in range(100):
        X_train, X_test, y_train, y_test = train_test_split( 
                    X, y, test_size = 0.1, random_state=i) 
        

        knn = KNeighborsClassifier(n_neighbors=j) 
        
        knn.fit(X_train, y_train) 
        # Predict on dataset which model has not seen before 
        n= knn.score(X_test, y_test)
        res.append(n)
        m = m+n
        # print("Accuracy at",i ,"=" , n)
    # print(sorted(res))
    print("Average accuracy for 100 runs when number of nearest neighbour is",j,"  =", m/100)