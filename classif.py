	#https://www.tensorflow.org/tutorials/load_data/images
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import glob
import cv2 
# in : 
# /home/kali/Bureau/PFE/projetGit/AuthentificationBiometrique
# python3 -m http.server  
# only 1st time to create the keras dataset locally (at /home/kali/.keras/datasets/flower_photos)
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import openpyxl 

import pathlib

import argparse

img_height = 320
img_width = 120
ds_size=756
nb_classes=126


parser = argparse.ArgumentParser()

parser.add_argument('-g','--graph',
dest='graph',
help='Displays learing graph of the model, False by default')

parser.add_argument('-v','--verbose',
dest='verb',
help='Displays some informations such as model\'s summary, number of classes. False by default')

args = parser.parse_args()


def setup(epochs):

	dataset_url="http://0.0.0.0:8000/BDD_FingerVeins.tar.gz"
	data_dir = tf.keras.utils.get_file(fname='DB_clean', origin=dataset_url, untar=True)
	data_dir = pathlib.Path(data_dir)
	
	#image_count = len(list(data_dir.glob('*/*.bmp')))
	#print("images used in dataset: ",image_count)
	
	imgs = list(data_dir.glob('*/*.bmp')) # images dans les sous dossiers correspondant à leur classe
	
	batch_size = 32

	train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

	val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


#	for image_batch, labels_batch in train_ds:
#		print(image_batch.shape)
#		print(labels_batch.shape)
#		break

	class_names = train_ds.class_names
	#print("Classes: ")
	#print(class_names)
	
	#print(train_ds)
##	plt.figure(figsize=(10, 10))
#	for images, labels in train_ds.take(1):
#		for i in range(9):
#			ax = plt.subplot(3, 3, i + 1)
#			plt.imshow(images[i].numpy().astype("uint8"))
#			plt.title(class_names[labels[i]])
##			plt.axis("off")


#	for image_batch, labels_batch in train_ds:
#		print(image_batch.shape)
#		print(labels_batch.shape)
#		break

	
# https://www.tensorflow.org/tutorials/load_data/images#standardize_the_data

	
	# normalization
#	normalization_layer = tf.keras.layers.Rescaling(1./255)
	
#	normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
	
#	print(normalized_ds)
#	print("------")	
#
#	image_batch, labels_batch = next(iter(normalized_ds))
#	print(image_batch, labels_batch )
#	first_image = image_batch[0]
	# Notice the pixel values are now in `[0,1]`.
#	print(np.min(first_image), np.max(first_image))


# 0.0 0.96902645


	model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(nb_classes) # nb couches sorties/classes
])


	model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

	history = model.fit(train_ds,validation_data=val_ds,epochs=epochs)

	if args.verb:
		model.summary()

	test_images=val_ds
	probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
	predictions = probability_model.predict(test_images)

	#for b in range(5):
	##for images, labels in test_images.take(1):
		#for i in range(1):
				#ax = plt.subplot(3, 3, i + 1)
				#plt.imshow(images[i].numpy().astype("uint8"))
				#plt.title(class_names[labels[i]])
			#print(i,": ",class_names[labels[i]])
				#plt.axis("off")


	#print(predictions[0])
	#z=np.argmax(predictions[0])
	#print(z)
	#print(class_names[z])

	#print("-----------")
	#print(predictions[70])	
#	#print(len(predictions[150]))
	#y=np.argmax(predictions[70])
	#print(y)
	#print(class_names[y])

	#if args.graph:
	acc = history.history['accuracy']
	val_acc = history.history['val_accuracy']
	
	loss = history.history['loss']
	val_loss = history.history['val_loss']

	epochs_range = range(epochs)

	plt.figure(figsize=(8, 8))
	plt.subplot(1, 2, 1)
	plt.plot(epochs_range, acc, label='Training Accuracy')
	plt.plot(epochs_range, val_acc, label='Validation Accuracy')
	plt.legend(loc='lower right')
	plt.title('Training and Validation Accuracy')
	
	plt.subplot(1, 2, 2)
	plt.plot(epochs_range, loss, label='Training Loss')
	plt.plot(epochs_range, val_loss, label='Validation Loss')
	plt.legend(loc='upper right')
	plt.title('Training and Validation Loss')
	plt.show()	

		
	## test
	#sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
	#sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)
	#sunflower_path = "/home/kali/.keras/datasets/DB_clean/001left_index/001left_index_1.bmp"
	
	#img = tf.keras.utils.load_img(impath, target_size=(img_height, img_width))
	#img_array = tf.keras.utils.img_to_array(img)
	#img_array = tf.expand_dims(img_array, 0) # Create a batch
	#predictions = model.predict(img_array)
	#score = tf.nn.softmax(predictions[0])
	#print("matrice de prediction",predictions[0])

	#print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)))
	#print(np.max(score))

	return model, class_names
setup(7)





# Realize nb_iter builds of the model, each with nb_epochs epochs. 
#Each build is striclty independant from the other, it permits to verify that the model generated is almost always the same
def excel_models(nb_iter,nb_epochs):
  
	# each model will test the following image
	impath = "/home/kali/.keras/datasets/004left_index_1.bmp"  
	targ=impath.split("/")[-1]
	targ=targ[:-6]
	
	target=[]
	result=[]
	confidence=[]
	correct=[]

	for i in range(nb_iter):
		img = tf.keras.utils.load_img(impath, target_size=(img_height, img_width))
		img_array = tf.keras.utils.img_to_array(img)
		img_array = tf.expand_dims(img_array, 0)
		m,classes=setup(nb_epochs)
		predictions = m.predict(img_array)
		score = tf.nn.softmax(predictions[0])
		res=classes[np.argmax(score)]
		boolean="false"
		if str(targ)==str(res):
			boolean="true"
			print("ok T")
		print("{} most likely belongs to {} class with a {:.2f} % confidence => {}".format(targ,res, 100 * np.max(score),boolean))
		target.append(targ)
		result.append(res)
		confidence.append("{:.2f}".format(100 * np.max(score)))
		correct.append(boolean)				

	# writing datas on excel
	wb = openpyxl.Workbook()  
	sheet = wb.active 

	ca = sheet.cell(row = 1, column = 1)
	ca.value = "Target class"
	cb = sheet.cell(row = 1, column = 2) 
	cb.value = "Predicted class"	
	cc = sheet.cell(row = 1, column = 3) 
	cc.value = "Confidence %"
	cd = sheet.cell(row = 1, column = 4) 
	cd.value = "Correctness"
	ce = sheet.cell(row = 1, column = 5)
	ce.value = "Iteration"
	for i in range(2,nb_iter+2):
		print("i: ",i)
		c1 = sheet.cell(row = i, column = 1) 
		c1.value = target[i-2]
		c2 = sheet.cell(row = i, column = 2) 
		c2.value = result[i-2]
		c3 = sheet.cell(row = i, column = 3) 
		s=confidence[i-2]#.replace(".",",")
		c3.value = float(s)
		c4 = sheet.cell(row = i, column = 4) 
		c4.value = correct[i-2]
		c5 = sheet.cell(row = i, column = 5) 
		c5.value = i-2	

	
	cmt = sheet.cell(row = nb_iter+2, column = 4)
	cmt.value = 'Average confidence'
	cm = sheet.cell(row = nb_iter+2, column = 5)	
	cm.value = '=AVERAGE(C2:C{})'.format(nb_iter+2)

	ctf = sheet.cell(row = nb_iter+3, column = 4)
	ctf.value = 'True/False ratio'
	tfr = sheet.cell(row = nb_iter+2, column = 4)
	for e in correct:
		if e=="true":
			ratio+=1
	tfr.value = '={}/{}'.format(ratio,nb_iter)



	wb.save("/home/kali/Bureau/PFE/out-{}models-{}epochs.xlsx".format(nb_iter,nb_epochs)) 


#excel_models(5,5)




