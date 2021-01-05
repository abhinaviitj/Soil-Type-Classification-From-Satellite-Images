import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from skimage.feature import greycomatrix, greycoprops 

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
            f = list(contrast)+list(dissimilarity) +list(homogeneity)+list(energy)
            feat.append(f)
        img = cv2.imread(os.path.join(folder,filename[i+1]),0)
        img = np.array(img, dtype = np.uint8)
        gCoMat = greycomatrix(img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels = None)
        contrast = greycoprops(gCoMat, prop='contrast')
        dissimilarity = greycoprops(gCoMat, prop='dissimilarity')
        homogeneity = greycoprops(gCoMat, prop='homogeneity')
        energy = greycoprops(gCoMat, prop='energy')
        # correlation = greycoprops(gCoMat, prop='correlation')
        f = list(contrast)+list(dissimilarity) +list(homogeneity)+list(energy)
        feat.append(f)
        fet = feat[0]+feat[1]+feat[2]+feat[3]
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




print("all features calculated")
# print()

all_images = np.float32(all_images)
print("ready for k means")

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS
compactness,labels,centers = cv2.kmeans(all_images, 4, None, criteria, 10, flags)
f=open("res_natural&thermal_kmeans.txt", "a+")
f.write("\n")

lst = ["alluvial","black","desert","red"]
f.write("result when clustering is on only natural colour and thermal band:\n")
for j in range(4):
    k=j*20
    f.write("clustering result of "+lst[j]+" soil\n")
    clus_dict={}
    for h in range(4):
        clus_dict[h]=0
    for i in range(20):
        clus_dict[labels[k+i][0]]+=1
    for key in sorted(clus_dict.keys()):
        f.write("Cluster "+str(key)+" : "+str(clus_dict[key])+"\n")
print("result done")