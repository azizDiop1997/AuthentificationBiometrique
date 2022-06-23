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


parser.add_argument('-r','--reducedBDD',
required=True,
dest='redu',
help='Number of images to create')


args = parser.parse_args()

def augment() : 
	if args.redu=="true":
		p = Augmentor.Pipeline(source_directory="../BDD_FingerVeins_reduced/", output_directory="../database/")
	else:
		p = Augmentor.Pipeline(source_directory="../BDD_FingerVeins/", output_directory="../database/")
	p.random_distortion(probability=0.2,grid_width=6, grid_height=6, magnitude=10)
	p.zoom(probability=0.2, min_factor=1.0, max_factor=1.8)
	p.shear(probability=0.55, max_shear_left=2, max_shear_right=8)
	p.crop_random(probability=0.25, percentage_area=0.85)
	p.flip_random(probability=0.25)
	#p.resize(probability=1, width=120, height=320)
	p.sample(int(args.n))
	
augment()
