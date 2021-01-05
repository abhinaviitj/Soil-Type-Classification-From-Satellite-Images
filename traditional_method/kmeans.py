import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from skimage.feature import greycomatrix, greycoprops 

def return_features(folder) :
    features = []
    i = 1
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        # print(os.path.join(folder,filename))
        img = np.array(img, dtype = np.uint8)
        feat=[]
        # img = np.transpose(img,(2,0,1))
        # img = img.reshape(3,-1)
        for j in range(len(img)):
            # res = []
            # for k in range(len(img)):
            #     lst = []
            #     for l in img[k]:
            #         lst.append(l[j])
            #     res.append(lst)
            # if j==1:
            #     continue
            gCoMat = greycomatrix(img[j], [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels = None)
            contrast = greycoprops(gCoMat, prop='contrast')
            dissimilarity = greycoprops(gCoMat, prop='dissimilarity')
            homogeneity = greycoprops(gCoMat, prop='homogeneity')
            energy = greycoprops(gCoMat, prop='energy')
            # correlation = greycoprops(gCoMat, prop='correlation')
            # print(contrast)
            # print(dissimilarity)
            # print(homogeneity)
            # print(energy)
            # print(j)
            # feat.append( contrast + dissimilarity + homogeneity + energy)
            # f =np.concatenate(contrast.reshape(4),dissimilarity.reshape(4))
            # f=np.concatenate(feat,homogeneity)
            # f= np.concatenate(feat,energy)
            f = list(contrast)+list(dissimilarity) +list(homogeneity)+list(energy)
            feat.append(f)
        fet = feat[0]+feat[1]+feat[2]
        fet = np.array(fet)
        # print(fet)   
        features.append(fet)
        print(i)
        i=i+1
        if i == 2:
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
print("red done")
all_images = all_images + images




print("all features calculated")
# print()
# all_images = np.array(all_images)
all_images = np.float32(all_images)
print("ready for k means")

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS
compactness,labels,centers = cv2.kmeans(all_images, 4, None, criteria, 10, flags)

print("labels : ")
i = 0
while i < 20 :
    print(labels[i], labels[i+25], labels[i+50], labels[i +75])
    i = i + 1