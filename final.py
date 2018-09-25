import pandas as pd
import numpy as np
reader=pd.read_csv("train.csv")
low_memory=False
X_train=[]
y_train=[]
final_x=[]
final_y=[]
print reader.index[0]
for index,row in reader.iterrows():
	for i in range(1,785):
		X_train.append(row[i])
		
	#x_train=np.array(X_train)
	#x_train=x_train.reshape(28,28)	
	final_x.append(X_train)
	final_y.append(row[0])
	X_train=[]
print final_x[0],final_y[0]
final_x_train=np.array(final_x)
final_y_train=np.array(final_y)
from sklearn.svm import LinearSVC
X_test=[]
from sklearn.datasets import make_classification
clf = LinearSVC()
clf.fit(final_x_train,final_y_train)
reader2=pd.read_csv("test.csv")
j=1
fi=open("submission.csv","w")
for index,row in reader2.iterrows():
	for i in range(0,784):
		X_test.append(row[i])
		
#	x_test=np.array(X_test)
#	x_test.reshape(1,-1)
		
	y_predicted=clf.predict([X_test])
	X_test=[]
	print y_predicted[0]
	fi.write(str(j))
	fi.write(",")
	fi.write(str(y_predicted[0]))
	
	fi.write('\n')
	j=j+1



