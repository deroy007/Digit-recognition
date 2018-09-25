from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from skimage.color import rgb2gray
from skimage import data
from sklearn import datasets, svm, metrics
def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        # no deskewing needed. 
        return img.copy()
    # Calculate skew based on central momemts. 
    skew = m['mu11']/m['mu02']
    # Calculate affine transform to correct skewness. 
    M = np.float32([[1, skew, -0.5*20*skew], [0, 1, 0]])
    # Apply affine transform
    img = cv2.warpAffine(img, M, (20, 20), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

#img = cv2.imread('1/13.jpg')
im1 = cv2.imread("1/11.jpg")
im2 = cv2.imread("1/12.jpg")
im3 = cv2.imread("2/21.jpg")
im4 = cv2.imread("2/22.jpg")
im5 = cv2.imread("1/13.jpg")
im6 = cv2.imread("2/23.jpg")
#im1=rgb2gray(im1)
#im2=rgb2gray(im2)
#im3=rgb2gray(im3)
#im4=rgb2gray(im4)
#im5=rgb2gray(im5)
#im6=rgb2gray(im6)
im1= cv2.resize(im1, (1000,1000),interpolation=cv2.INTER_CUBIC)
im2= cv2.resize(im2, (1000,1000),interpolation=cv2.INTER_CUBIC)
im3= cv2.resize(im3, (1000,1000),interpolation=cv2.INTER_CUBIC)
im4= cv2.resize(im4, (1000,1000),interpolation=cv2.INTER_CUBIC)
im5= cv2.resize(im5, (1000,1000),interpolation=cv2.INTER_CUBIC)
im6= cv2.resize(im6, (1000,1000),interpolation=cv2.INTER_CUBIC)
#im1=im1.reshape(1000000)
#im2=im2.reshape(1000000)
#im3=im3.reshape(1000000)
#im4=im4.reshape(1000000)
#im5=im5.reshape(1000000)
#im6=im6.reshape(1000000)     
#gray = rgb2gray(img) 
#print img.shape,gray.shape
#plt.plot(gray) 
#plt.show()  
#gray_deskew=deskew(gray)
#plt.plot(gray_deskew) 
#plt.show()  
#print gray_deskew
#gray_deskew=gray_deskew.reshape(20,20)
#img = data.astronaut()
winSize = (20,20)
blockSize = (10,10)
blockStride = (5,5)
cellSize = (10,10)
nbins = 9
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
signedGradients = True
from sklearn import linear_model, datasets
hog = cv2.HOGDescriptor()
descriptor1 = hog.compute(im1)
descriptor2 = hog.compute(im2)
descriptor3 = hog.compute(im3)
descriptor4 = hog.compute(im4)
descriptor5 = hog.compute(im5)
descriptor6 = hog.compute(im6)
#np.vstack(descriptor1)
#descriptor1.reshape(49064400,)
#descriptor2.reshape(49064400,)
#descriptor3.reshape(49064400,)
#descriptor4.reshape(49064400,)
#descriptor5.reshape(49064400,)
#descriptor6.reshape(49064400,)
#print descriptor1.shape
#l=[]
#l=[descriptor1,descriptor2,descriptor3,descriptor4]
#p=np.array(l)
#print p.shape
#print descriptor1,descriptor2
descriptor1=descriptor1.reshape(-1)
descriptor2=descriptor2.reshape(-1)
descriptor3=descriptor3.reshape(-1)
descriptor4=descriptor4.reshape(-1)
descriptor5=descriptor5.reshape(-1)
descriptor6=descriptor6.reshape(-1)
l=[]
l=[descriptor1,descriptor2,descriptor3,descriptor4]
X_plot=np.array(l)
np.vstack(X_plot)
X_plot.reshape(4,-1)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#descriptor1=descriptor1.reshape(1,-1)
y_plot=np.array(['1','1','2','2'])
y_plot.reshape(4,-1)
from sklearn.svm import LinearSVC

from sklearn.datasets import make_classification
clf = LinearSVC()
clf.fit(X_plot, y_plot)
#print pca.singular_values_
#print g.shape
#y_plot=np.array(['1','1','2','2'])
#y_plot.reshape(4,-1)
#g=descriptor1.flatten('F')
#print g
#classifier = svm.SVC(gamma=0.001)
#classifier.fit(X_plot,y_plot)
#predicted=classifier.predict(descriptor5)
#print predicted
#print descriptor1.shape,descriptor2.shape
#print X_plot,y_plot

