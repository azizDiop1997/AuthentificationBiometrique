import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2 
import Augmentor
import glob
import os

parser = argparse.ArgumentParser()

parser.add_argument('-n','--numImg',
required=True,
dest='n',
help='Number of images to create')

args = parser.parse_args()

def augment() :
	p = Augmentor.Pipeline(source_directory="../BDD_FingerVeins/", output_directory="../database/")
	p.random_distortion(probability=0.4,grid_width=6, grid_height=6, magnitude=8)
	p.zoom(probability=0.2, min_factor=1.1, max_factor=1.6)
	p.shear(probability=0.6, max_shear_left=5, max_shear_right=5)
	p.crop_random(probability=0.4, percentage_area=0.8)
	p.flip_random(probability=0.4)
	#p.resize(probability=1, width=120, height=320)
	p.sample(int(args.n))
	
augment()

# rename & add images to dataset directories
def rename():
	images = glob.glob('../database/*.bmp')

	for img in images: 
		short=img.split(".")[2]	
		#print(short)
		short=short.split("original_")[-1]	
		#print(short)
		s="../database/"+short+"_modif"+".bmp"
		os.system('mv {} {}'.format(img,s))
		#print(short,"--",s)

		s2=s.split("_mod")[0].split("/")[-1]
		
		s2="/home/kali/.keras/datasets/DB_augmented/"+s2+"_modif"+".bmp"
		print(s2)
		os.system('mv {} {}'.format(s,s2))
		# pre traitement
		os.system('python3 /home/kali/Bureau/PFE/newGit/AuthentificationBiometrique/pycode_traitement_images/traitement_imagesv1.py -i {} -o {}'.format(s2,s2))
		print("mv to {}".format(short[:-2]))
		os.system('mv {} /home/kali/.keras/datasets/DB_augmented/{}/'.format(s2,short[:-2]))

rename()	
