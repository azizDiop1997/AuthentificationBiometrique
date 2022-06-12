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

def setup():

	dataset_url="http://0.0.0.0:8000/BDD_FingerVeins.tar.gz"
	data_dir = tf.keras.utils.get_file(fname='BDD_FingerVeins', origin=dataset_url, untar=True)
	print("data_dir: ",data_dir)
	data_dir = pathlib.Path(data_dir)
	#image_count = len(list(data_dir.glob('001left*/*.bmp')))
	#image_count = len(list(data_dir.glob('*.bmp'))) #756 all in dir
	#print("image_count: ",image_count)	
	
	imgs = list(data_dir.glob('*/*.bmp')) # trier toutes les images dans des sous dossiers
	x=2
	print(PIL.Image.open(str(imgs[x])))
	print(str(imgs[x]))
	PIL.Image.open(str(imgs[x])) ## ???
	print("---",str(imgs[7]))
	print("---",str(imgs[3]))

	for i in range (0,756):
		print(i)
		if ".bmp" in str(imgs[i]):
			image = cv2.imread(str(imgs[i]))
			height, width = image.shape[:2]
			#print(height, width,str(imgs[i]))
			if((height, width)!=(240,320)):
				print("error",str(imgs[i]),height, width)
				#exit(-1)
		#else:
			#print(i," rep ",str(imgs[i]))
	
	print("end")

	batch_size = 32
	img_height = 320
	img_width = 240


	train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

	print(train_ds)

	val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


	class_names = train_ds.class_names
	print(class_names)

	plt.figure(figsize=(10, 10))
	for images, labels in train_ds.take(1):
		for i in range(9):
			ax = plt.subplot(3, 3, i + 1)
			plt.imshow(images[i].numpy().astype("uint8"))
			plt.title(class_names[labels[i]])
			plt.axis("off")
			print('ok')

#	for image_batch, labels_batch in train_ds:
#		print(image_batch.shape)
#		print(labels_batch.shape)
#		break

setup()


















