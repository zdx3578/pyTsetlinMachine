from pyTsetlinMachine.tm import MultiClassConvolutionalTsetlinMachine2D
import numpy as np
from time import time

from keras.datasets import cifar10
import datetime


(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
Y_train=Y_train.reshape(Y_train.shape[0])
Y_test=Y_test.reshape(Y_test.shape[0])


# X_train = X_train[0:200]
# Y_train = Y_train[0:200]

# train_filter = np.where((Y_train == 0 ) | (Y_train == 1))
# test_filter = np.where((Y_test == 0) | (Y_test == 1))
# X_train, Y_train = X_train[train_filter], Y_train[train_filter]
# X_test, Y_test = X_test[test_filter], Y_test[test_filter]

# X_train = np.dot(X_train[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint32)
# X_test = np.dot(X_test[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint32)


unpackbit=4

#tm = MultiClassConvolutionalTsetlinMachine2D(8000, 200, 10.0, (7, 7))
#tm = MultiClassConvolutionalTsetlinMachine2D(1000, 25*100, 10.0, (10, 10), weighted_clauses=True)
cifa10encode = 12 #  12 is 32 32 12;  3 is   64 64 3;;
tm = MultiClassConvolutionalTsetlinMachine2D(1000, 25*100, 6.0, (10,10), number_of_state_bits=8,append_negated=True,stride=4,unpackbit=unpackbit/2,weighted_clauses=True,cifa10encode=cifa10encode)
print(datetime.datetime.now())

print("\nAccuracy over 20 epochs:\n")
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
	#print()
