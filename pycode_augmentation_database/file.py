import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2 
import Augmentor

# parser = argparse.ArgumentParser()

# parser.add_argument('-i','--image',
# required=True,
# dest='image',
# help='select image of the finger'
# )

# args = parser.parse_args()
# img = cv2.imread(args.image,0)

def augment() :
	p = Augmentor.Pipeline(source_directory="../BDD_FingerVeins/", output_directory="../database/")
	
	p.random_distortion(probability=0.2,grid_width=4, grid_height=4, magnitude=8)
	p.zoom(probability=0.2, min_factor=1.1, max_factor=1.6)
	p.shear(probability=0.2, max_shear_left= 25, max_shear_right=1)
	p.crop_random(probability=0.2, percentage_area=0.8)
	p.flip_random(probability=0.2)
	p.resize(probability=1, width=120, height=320)
	p.sample(10000)
	
augment()
