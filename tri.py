import glob
import os

images = glob.glob('DB_clean/*.bmp')
print(images)
# python3 tri.py | grep bmp | wc

directories=[]
x=9 # longueur chemin avant nom image
# or use split
for img in images: 
	print(img)
	short=img[x:-6]
	print(short)
	if img[x:-6] not in directories:
		os.system("mkdir {}".format(img[x:-6]))
	os.system("mv {} {}".format(img,img[x:-6]))



os.system('mv 0*/ DB_clean')

# to adapt: / or use keras untar
#os.system('cp -r BDD_FingerVeins /home/kali/.keras/datasets')


