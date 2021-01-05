import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from skimage.feature import local_binary_pattern, greycomatrix, greycoprops

def return_glcm(folder) :
    images = []
    i = 1
    filename = os.listdir(folder)
    for i in range(0,40,2):
        fet=[]
        img = cv2.imread(os.path.join(folder,filename[i+1]),0)
        glcm = greycomatrix(img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels = None)
        glcm = glcm[:,:,0,0]
        fet=fet+list(glcm)
        glcm = np.array(fet)
        if img is not None:
            images.append(glcm)
    return images

all_images = []

images = return_glcm("alluvial")
print("alluvial done")
all_images = all_images + images

images = return_glcm("black")
print("black done")
all_images = all_images + images

images = return_glcm("desert")
print("desert done")
all_images = all_images + images

images = return_glcm("red")
print("red done")
all_images = all_images + images


print("all GLCM calculated")
print()

all_images = np.float32(all_images)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

flags = cv2.KMEANS_RANDOM_CENTERS

compactness,labels,centers = cv2.kmeans(all_images, 4, None, criteria, 10, flags)
f=open("res_th_c_kmeans.txt", "a+")
f.write("\n")

lst = ["alluvial","black","desert","red"]
f.write("result when clustering is on only thermal band with color as feature:\n")
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