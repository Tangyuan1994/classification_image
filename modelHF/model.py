import numpy as np
import matplotlib.pyplot as plt
import sys
import caffe
import os

#1.Set caffe, GPU and basic setting
#selon votre caffe_root
caffe_root = 'home/tangyuan/caffe/'
sys.path.insert(0, caffe_root+'python')
caffe.set_device(0)
caffe.set_mode_gpu()
#figuresize
plt.rcParams['figure.figsize'] = (10,10)
#methode d'interpolation
plt.rcParams['image.interpolation'] = 'nearest'
#Grayscale space
plt.rcParams['image.cmap'] = 'gray'


#2.Build network by deploy_prototxt and caffemodel
model_def = 'deploy_gender.prototxt'
model_weights = 'gender_net.caffemodel'
net = caffe.Net(model_def, model_weights, caffe.TEST)

#3.pre-processing of images
mean_filename = './mean.binaryproto'
proto_data = open(mean_filename, 'r').read()
a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
mean = caffe.io.blobproto_to_array(a)[0].mean(1).mean(1)
transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  
transformer.set_mean('data', mean)
transformer.set_raw_scale('data', 255);
#swap channels from RGB to BGR
transformer.set_channel_swap('data', (2,1,0)) 
net.blobs['data'].reshape(1,3,227,227)


#4.read images and predict the sex
result=[]
for i in range(1,202600):
#because the name of images is look like 000001.jpg (padding with 0)
	h = "%06d" % i
	image = caffe.io.load_image('./'+str(h)+'.jpg')
	transformed_image = transformer.preprocess('data', image)
	net.blobs['data'].data[...] = transformed_image
#hommes est 1, femme est -1
	labels = [1,-1]
#classify the output
	output = net.forward()
	output_prob = output['prob'][0]
	print output_prob
	print labels[output_prob.argmax()]
	result.append(labels[output_prob.argmax()])

#5.calcul accuracy by comparing with labels
#je nomme list_attr_celeba.txt comme attibu.txt
fp=open(r"list_attr_celeba.txt")
#For stocking my prediction of label
f = open("prediction.txt", 'w+')  
label=[]
label_male=[]
#split by " "
for linea in fp.readlines():
    linea=linea.split(" ")
    label.append(linea)

for element in label:
	while '' in element:
    		element.remove('')
#read the 21th colone as the atteibute "male"
for element in label:
	label_male.append(element[21])
#remove the first element in label_male beacause is a the first line in is all the names of the attributes.
label_male=label_male[1:]

#save my predictions in the ficher prediction.txt
prediction=[]
for element in result:
	prediction.append(element)
for element in prediction:
	print >>f,element

#compare my prediction with the labels in list_attr_celeba.txt
count=0
for i in range(0,202599):
	if label_male[i]==str(prediction[i]):
		count+=1

#calculer accuracy
print count
accuracy=float(count)/202599
print accuracy
fp.close()
