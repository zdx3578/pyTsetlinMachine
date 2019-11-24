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


def clip_bits(X_train_rgb, num):
	def convert_img(X_train,num):
		def n2b(x_0):
			data_x = []

			for x in x_0.T:
				x_ = x.reshape(len(x), 1).astype(dtype=np.uint8)  # reshape to 1d array
				x_1 = np.unpackbits(x_, axis=1)[:, :num]
				x_2 = np.vstack([x.reshape(2, 2) for x in x_1])
				data_x.append(x_2)
				x_3 = np.hstack(data_x)  # conver 2d to 2d bit
			return x_3
		x_4 = np.stack([n2b(x) for x in X_train.T], axis=2)  # apply to rgb img
		return x_4

	X_b = np.stack([convert_img(X_train,num) for X_train in X_train_rgb], axis=0)
	return X_b


X_train = clip_bits(X_train, self.unpackbit*self.unpackbit)
X_test = clip_bits(X_test, self.unpackbit*self.unpackbit)


unpackbit=4
# if unpackbit%2!=0:
# 	break

#tm = MultiClassConvolutionalTsetlinMachine2D(8000, 200, 10.0, (7, 7))
tm = MultiClassConvolutionalTsetlinMachine2D(8000, 250, 12.0, (8, 8),stride=1,unpackbit=unpackbit/2)

print("\nAccuracy over 40 epochs:\n")
for i in range(40):
	start = time()
	print(datetime.datetime.now())
	tm.fit(X_train, Y_train, epochs=1, incremental=True)
	stop = time()
	print(datetime.datetime.now())
	result = 100*(tm.predict(X_test) == Y_test).mean()
	print("#%d Accuracy: %.2f%% (%.2fs)" % (i+1, result, stop-start))

	result2 = 100 * (tm.predict(X_train) == Y_train).mean()
	print("#%d Accuracy: %.2f%% (%.2fs)" % (i + 1, result2, stop - start))

 	print()
