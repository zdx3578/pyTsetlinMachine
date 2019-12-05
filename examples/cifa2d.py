from pyTsetlinMachine.tm import MultiClassConvolutionalTsetlinMachine2D
import numpy as np
from time import time

from keras.datasets import cifar10
import datetime


(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
Y_train=Y_train.reshape(Y_train.shape[0])
Y_test=Y_test.reshape(Y_test.shape[0])



train_filter = np.where((Y_train == 0 ) | (Y_train == 1))
test_filter = np.where((Y_test == 0) | (Y_test == 1))
X_train, Y_train = X_train[train_filter], Y_train[train_filter]
X_test, Y_test = X_test[test_filter], Y_test[test_filter]
print(" 2 class train ")


# X_train = X_train[0:200]
# Y_train = Y_train[0:200]
#
# X_test = X_test[0:200]
# Y_test = Y_test[0:200]


cifa10encode = 12 #  12 is 32 32 12;  3 is   64 64 3;;
bit=1

num=4 ## unpackbit count= -num

#cifa10encode = 3 #  12 is 32 32 12;  3 is   64 64 3;;
#bit=2

tm = MultiClassConvolutionalTsetlinMachine2D(26000, 100*100, 6.0, (8*bit,8*bit),weighted_clauses=True,cifa10encode=cifa10encode,num = num)
print(datetime.datetime.now())

print("\nAccuracy over 20 epochs:\n")
for i in range(40):
	start = time()
	#print(datetime.datetime.now())
	tm.fit(X_train, Y_train, epochs=1, incremental=True)
	stop = time()
	#print(datetime.datetime.now(),"predict")
	result = 100*(tm.predict(X_test) == Y_test).mean()
	print("#%d Accuracy: %.2f%% (%.2fs)" % (i+1, result, stop-start))





