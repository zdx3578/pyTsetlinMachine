from pyTsetlinMachineParallel.tm import MultiClassConvolutionalTsetlinMachine2D
import numpy as np
from time import time
import cv2
from keras.datasets import *





(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

# X_train = X_train[0:200]
# Y_train = Y_train[0:200]
# X_test = X_test[0:200]
# Y_test = Y_test[0:200]

Y_train=Y_train.reshape(Y_train.shape[0])
Y_test=Y_test.reshape(Y_test.shape[0])

#train_filter = np.where((Y_train == 0 ) | (Y_train == 1))
#test_filter = np.where((Y_test == 0) | (Y_test == 1))
#X_train, Y_train = X_train[train_filter], Y_train[train_filter]
#X_test, Y_test = X_test[test_filter], Y_test[test_filter]
#print(" 2 class train ")

for i in range(X_train.shape[0]):
	for j in range(X_train.shape[3]):
		X_train[i,:,:,j] = cv2.adaptiveThreshold(X_train[i,:,:,j], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)

for i in range(X_test.shape[0]):
	for j in range(X_test.shape[3]):
		X_test[i,:,:,j] = cv2.adaptiveThreshold(X_test[i,:,:,j], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)
	#X_test[i,:] = cv2.adaptiveThreshold(X_test[i], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)



X_train2 = X_train[0:2000]
Y_train2 = Y_train[0:2000]



tm = MultiClassConvolutionalTsetlinMachine2D(12000, 150*100, 4, (4, 4), weighted_clauses=True)

print("\nAccuracy over 120 epochs:\n")
for i in range(120):
	start = time()
	tm.fit(X_train, Y_train, epochs=1, incremental=True)
	stop = time()
	
	result = 100*(tm.predict(X_test) == Y_test).mean()
	
	print("#%d test  Accuracy: %.2f%% (%.2fs)" % (i+1, result, stop-start))
	result2 = 100*(tm.predict(X_train2) == Y_train2).mean()
	print("#%d train  Accuracy: %.2f%% (%.2fs)" % (i+1, result2, stop-start))	
