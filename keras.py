import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.optimizers import Adam
#from keras.layers.normalization import BatchNormalization
#from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
#from tensorflow.python.keras.layers.advanced_activations import LeakyReLU 
#from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

reader=pd.read_csv("train.csv")
low_memory=False
X_train=[]
y_train=[]
final_x=[]
final_y=np.zeros([42000,10],dtype=int)
print reader.index[0]
for index,row in reader.iterrows():
	for i in range(1,785):
		X_train.append(row[i])
	x_train=np.array(X_train).reshape(28,28)
	#x_train=np.array(X_train)
	#x_train=x_train.reshape(28,28)	
	final_x.append(x_train)
	c=int(row[0])
	#print c,type(c)
	if c==0:
		final_y[index][0]=1
	elif c==1:
		final_y[index][1]=1
	elif c==2:
		final_y[index][2]=1
	elif c==3:
		final_y[index][3]=1
	elif c==4:
		final_y[index][4]=1
	elif c==5:
		final_y[index][5]=1
	elif c==6:
		final_y[index][6]=1
	elif c==7:
		final_y[index][7]=1
	elif c==8:
		final_y[index][8]=1
	elif c==9:
		final_y[index][9]=1
	X_train=[]
	print final_y[index],index
#print final_y[0],final_y[1000]
final_x_train=np.array(final_x)
final_y_train=np.array(final_y)
final_x_train/=255
final_x_train=final_x_train.reshape(42000,28,28,1)
#final_y_train=final_y_train.reshape(42000,1,10)
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(28,28,1)))
#model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
#model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3, 3)))
#model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
#model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

# Fully connected layer
model.add(Dense(512))
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))

model.add(Activation('softmax'))
# COMPILE

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.fit(final_x_train,final_y_train,epochs=5,batch_size=32)
X_test=[]
j=0
fi=open("submission.csv","w")
reader2=pd.read_csv("test.csv")
for index,row in reader2.iterrows():
	for i in range(0,784):
		X_test.append(row[i])
	x_test=np.array(X_test).reshape(28,28)
		
	#x_test=np.array(X_test)
	x_test=x_test.reshape(1,28,28,1)
		
	y_predicted=model.predict_classes(x_test)
	X_test=[]
	print y_predicted
	fi.write(str(j))
	fi.write(",")
	fi.write(str(y_predicted[0]))
	j=j+1
	fi.write("\n")
#fi.close()






	

