from pyTsetlinMachine.tm import MultiClassConvolutionalTsetlinMachine2D
import numpy as np
from time import time

from keras.datasets import cifar10
import datetime


(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

# X_train = X_train[0:2500]
# Y_train = Y_train[0:2500]

Y_train=Y_train.reshape(Y_train.shape[0])
Y_test=Y_test.reshape(Y_test.shape[0])

train_filter = np.where((Y_train == 0 ) | (Y_train == 1))
test_filter = np.where((Y_test == 0) | (Y_test == 1))

X_train, Y_train = X_train[train_filter], Y_train[train_filter]
X_test, Y_test = X_test[test_filter], Y_test[test_filter]



unpackbit=4
# if unpackbit%2!=0:
# 	break

#tm = MultiClassConvolutionalTsetlinMachine2D(8000, 200, 10.0, (7, 7))
tm = MultiClassConvolutionalTsetlinMachine2D(14000, 800, 22.0, (10,10), number_of_state_bits=10,stride=2,unpackbit=unpackbit/2)

print("\nAccuracy over 40 epochs:\n")
for i in range(40):
	start = time()
	print(datetime.datetime.now())
	tm.fit(X_train, Y_train, epochs=1, incremental=True)
	stop = time()
	#print(datetime.datetime.now())
	result = 100*(tm.predict(X_test) == Y_test).mean()
	print("#%d Accuracy: %.2f%% (%.2fs)" % (i+1, result, stop-start))

	#result2 = 100 * (tm.predict(X_train) == Y_train).mean()
	#print("#%d Accuracy: %.2f%% (%.2fs)" % (i + 1, result2, stop - start))
	#print()
