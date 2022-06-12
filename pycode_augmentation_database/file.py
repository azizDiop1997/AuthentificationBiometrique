import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2 
import tensorflow as tf
import Augmentor

# parser = argparse.ArgumentParser()

# parser.add_argument('-i','--image',
# required=True,
# dest='image',
# help='select image of the finger'
# )

# args = parser.parse_args()
# img = cv2.imread(args.image,0)

# resize and rescale
p = Augmentor.Pipeline("../BDD_FingerVeins/")
# p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
p.zoom(probability=1, min_factor=1.1, max_factor=1.6)
p.sample(5)