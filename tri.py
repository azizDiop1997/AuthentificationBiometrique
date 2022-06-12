import glob
import os

images = glob.glob('BDD_FingerVeins/*.bmp')
print(images)
# python3 tri.py | grep bmp | wc
directories=[]
for img in images: 
	print(img)
	short=img[16:-6]
	if img[16:-6] not in directories:
		os.system("mkdir {}".format(img[16:-6]))
	os.system("mv {} {}".format(img,img[16:-6]))



os.system('mv 0*/ BDD_FingerVeins/')

# to adapt: / or use keras untar
#os.system('cp -r BDD_FingerVeins /home/kali/.keras/datasets')


