import glob
import os

images = glob.glob('DB_clean/*.bmp')
print(images)
# python3 tri.py | grep bmp | wc

directories=[]
x=9 # longueur chemin avant nom image
# or use split
for img in images: 
	#print(img)
	short=img.split("/")
	short=short[-1]
	#print(short)
	#print(short[:-6])
	directory=short[:-6]
	if directory not in directories:
		os.system("mkdir {}".format(directory))
	os.system("mv {} {}".format(img,directory))

os.system('mv 0*/ DB_clean')

# to adapt: / or use keras untar
os.system('DB_clean /home/kali/.keras/datasets')


