import cv2
import numpy as np
from sklearn import datasets, svm, metrics
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

l=[]
im1 = cv2.imread("1/11.jpg")
im2 = cv2.imread("1/12.jpg")
im3 = cv2.imread("2/21.jpg")
im4 = cv2.imread("2/22.jpg")
im5 = cv2.imread("1/13.jpg")
im6 = cv2.imread("2/23.jpg")
print im1
im1=rgb2gray(im1)
im2=rgb2gray(im2)
im3=rgb2gray(im3)
im4=rgb2gray(im4)
im5=rgb2gray(im5)
im6=rgb2gray(im6)
im1= cv2.resize(im1, (1000,1000),interpolation=cv2.INTER_CUBIC)
im2= cv2.resize(im2, (1000,1000),interpolation=cv2.INTER_CUBIC)
im3= cv2.resize(im3, (1000,1000),interpolation=cv2.INTER_CUBIC)
im4= cv2.resize(im4, (1000,1000),interpolation=cv2.INTER_CUBIC)
im5= cv2.resize(im5, (1000,1000),interpolation=cv2.INTER_CUBIC)
im6= cv2.resize(im6, (1000,1000),interpolation=cv2.INTER_CUBIC)
im1=im1.reshape(1000000)
im2=im2.reshape(1000000)
im3=im3.reshape(1000000)
im4=im4.reshape(1000000)
im5=im5.reshape(1000000)
im6=im6.reshape(1000000)

#im1=im1.reshape(69192)
#im2=im2.reshape(69192)

#plt.imshow(im2, interpolation='nearest')
#plt.show()



#print im11.shape

#d=np.array()
#print im1.shape
#l.append(im1)
#l.append(im2)
#l.append(im3)
#l.append(im4)
#l.append(im5)
#l.append(im6)
#l.append(im7)
#l.append(im8)
#l.append(im9)
#l.append(im10)
#l.append(im11)
#l.append(im12)
#l.append(im13)
#l.append(im14)
#l.append(im15)
#l.append(im16)
#l.append(im17)
#l.append(im18)
#l.append(im19)
#l.append(im20)
#print im1
l=[im1,im2,im3,im4]
#col_list = []
#for f in l:
 #   Temp = np.load(f,mmap_mode='r')
 #   col_list.append(Temp[:,0])
#print col_list
X_plot=np.array(l)
np.vstack(X_plot)
#print X_plot
X_plot.reshape(4,-1)
#print len(l_array),type(l_array)
y_plot=['1','1','2','2']
y_plot=np.array(y_plot)
y_plot.reshape(4,-1)
classifier = svm.SVC(gamma=0.001)
print X_plot.shape,y_plot.shape
classifier.fit(X_plot,y_plot)
print X_plot.shape
                                                             ##Testing phase of the classifier##
#im4= cv2.resize(im4, (1000,1000),interpolation=cv2.INTER_CUBIC)
#im3=im3.reshape(3000000)
#im2=im2[0:59685]
im6=im6.reshape(1,-1)
#print im2.size
predicted = classifier.predict(im6)
print predicted


