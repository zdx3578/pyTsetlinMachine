from pyTsetlinMachine.tm import MultiClassConvolutionalTsetlinMachine2D
import numpy as np
from time import time

from keras.datasets import cifar10
import datetime


(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

#X_train = X_train.reshape(X_train.shape[0], 32*32*3)
#X_test = X_test.reshape(X_test.shape[0], 32*32*3)

# X_train = X_train[0:2500]
# Y_train = Y_train[0:2500]

Y_train=Y_train.reshape(Y_train.shape[0])
Y_test=Y_test.reshape(Y_test.shape[0])

unpackbit=9
# if unpackbit%2!=0:
# 	break

#tm = MultiClassConvolutionalTsetlinMachine2D(8000, 200, 10.0, (7, 7))
tm = MultiClassConvolutionalTsetlinMachine2D(28000, 550, 8.0, (12, 12), number_of_state_bits=9,stride=4,unpackbit=unpackbit/3)

print("\nAccuracy over 40 epochs:\n")
for i in range(40):
	start = time()
	print(datetime.datetime.now())
	tm.fit(X_train, Y_train, epochs=1, incremental=True)
	stop = time()
	print(datetime.datetime.now())
	result = 100*(tm.predict(X_test) == Y_test).mean()
	print("#%d Accuracy: %.2f%% (%.2fs)" % (i+1, result, stop-start))

	#result2 = 100 * (tm.predict(X_train) == Y_train).mean()
	#print("#%d Accuracy: %.2f%% (%.2fs)" % (i + 1, result2, stop - start))
	print()
