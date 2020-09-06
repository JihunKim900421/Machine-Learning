# train,t10k-images - first 16bytes : header, / 0x00 : magic number, 0x04 : num of images, 0x08 : rows, 0x12 : columns -> 28*28=784 pixcels
# train,t10k-labels - first 8byte : header, / after 8bytes -> each 1byte : label

import struct
import numpy as np
from sklearn import model_selection, svm, metrics
from PIL import Image
import matplotlib.pyplot as plt

# User defined Func for byte to array
def label2array(name) :
	with open(name + '.idx1-ubyte', 'br') as f:
		data = f.read()
		array_data = bytearray(data)

	num = int.from_bytes(bytes(array_data[4:8]), byteorder='big', signed=False)
	a = array_data[8:]
	b = list(a)

	final_array = np.array(b).reshape(num)

	print(final_array.shape)

	return final_array
def img2array(name) :
	with open(name+'.idx3-ubyte', 'br') as f:
		data = f.read()
		array_data = bytearray(data)

	num = int.from_bytes(bytes(array_data[4:8]), byteorder='big', signed=False)
	rows = int.from_bytes(bytes(array_data[8:12]), byteorder='big', signed=False)
	cols = int.from_bytes(bytes(array_data[12:16]), byteorder='big', signed=False)

	a = array_data[16:]
	b = list(a)
	final_array = np.array(b).reshape(num, rows*cols)
	print(final_array.shape)

	return final_array

# Binary to Image, saving 10ea
def bin2img(name) :
	with open(name+'.idx3-ubyte', 'br') as f:
		data = f.read()
		array_data = bytearray(data)

	# extract num of item, row, col
	num = int.from_bytes(bytes(array_data[4:8]), byteorder='big', signed=False)
	rows = int.from_bytes(bytes(array_data[8:12]), byteorder='big', signed=False)
	cols = int.from_bytes(bytes(array_data[12:16]), byteorder='big', signed=False)

	# Del Header
	bin_array = array_data[16:]
	int_list = list(bin_array)

	# Make img file (10ea)
	img = Image.new('L', (rows,cols))
	pixcels = rows * cols

	for i in range(10) :
		unit = bin_array[i*pixcels:((i+1)*pixcels)]
		img.putdata(unit)
		name = str(i + 1) + '-unit.png'
		img.save(name)

	img.show()
	print('Done')

# Sampling and Plot
def bin2plt(name) :
	with open(name+'.idx3-ubyte', 'br') as f:
		data = f.read()
		array_data = bytearray(data)

	# extract num of item, row, col
	num = int.from_bytes(bytes(array_data[4:8]), byteorder='big', signed=False)
	rows = int.from_bytes(bytes(array_data[8:12]), byteorder='big', signed=False)
	cols = int.from_bytes(bytes(array_data[12:16]), byteorder='big', signed=False)

	# Del Header
	bin_array = array_data[16:]
	int_list = list(bin_array)

	plt.show()

	fig = plt.figure()
	axs = []

	for i in range(25) :
		pix = np.array(int_list[(i*rows*cols):((i+1)*rows*cols)]).reshape(rows, cols)
		axs.append(fig.add_subplot(5,5,i+1))
		plt.imshow(pix, cmap='gray', interpolation='nearest')
		plt.axis('off')

	fig.tight_layout()
	plt.show()

bin2plt('train-images')

# filename string
test_labels = 't10k-labels'
train_labels = 'train-labels'
test_img = 't10k-images'
train_img = 'train-images'

# binary to array
train = img2array(train_img)
train_p = img2array(test_img)
label = label2array(train_labels)
label_p = label2array(test_labels)

# SVM Learning
clf = svm.SVC()
clf.fit(train, label)

# Predict
predict = clf.predict(train_p)

# Compare
ac_score = metrics.accuracy_score(label_p,predict)
cl_report = metrics.classification_report(label_p,predict)

# Reporting
print("Accuracy =",ac_score)
print("Report =")
print(cl_report)
