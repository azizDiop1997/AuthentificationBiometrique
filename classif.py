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

import pathlib


img_height = 320
img_width = 120
ds_size=756
nb_classes=126

def setup():

	dataset_url="http://0.0.0.0:8000/BDD_FingerVeins.tar.gz"
	data_dir = tf.keras.utils.get_file(fname='DB_clean', origin=dataset_url, untar=True)
	data_dir = pathlib.Path(data_dir)
	#image_count = len(list(data_dir.glob('001left*/*.bmp')))
	#image_count = len(list(data_dir.glob('*.bmp'))) #756 all in dir
	#print("image_count: ",image_count)	
	
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


	class_names = train_ds.class_names
	#print("Classes: ")
	#print(class_names)
	
	#print(train_ds)
	plt.figure(figsize=(10, 10))
	for images, labels in train_ds.take(1):
		for i in range(9):
			ax = plt.subplot(3, 3, i + 1)
			plt.imshow(images[i].numpy().astype("uint8"))
			plt.title(class_names[labels[i]])
			plt.axis("off")

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
# ML : https://www.tensorflow.org/tutorials/images/classification

	epochs=8
	history = model.fit(train_ds,validation_data=val_ds,epochs=epochs)


	model.summary()

	#print(str(imgs[10]))
	test_images=val_ds
	probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
	predictions = probability_model.predict(test_images)
	
	#print(str(test_images))
	#print(test_images)

	#for b in range(5):
	##for images, labels in test_images.take(1):
		#for i in range(1):
				#ax = plt.subplot(3, 3, i + 1)
				#plt.imshow(images[i].numpy().astype("uint8"))
				#plt.title(class_names[labels[i]])
			#print(i,": ",class_names[labels[i]])
				#plt.axis("off")

	#print(test_images[150])

	#print(str(test_images[0]))
	#print(str(test_images[150]))


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
	impath = "/home/kali/.keras/datasets/004left_index_1.bmp"

	img = tf.keras.utils.load_img(impath, target_size=(img_height, img_width))
	img_array = tf.keras.utils.img_to_array(img)
	img_array = tf.expand_dims(img_array, 0) # Create a batch

	predictions = model.predict(img_array)
	score = tf.nn.softmax(predictions[0])
	print("matrice de prediction",predictions[0])
	print(predictions[0])
	print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)))
	print(np.max(score))



setup()


















