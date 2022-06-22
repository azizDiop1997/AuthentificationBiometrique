import glob
import os

#images = glob.glob('DB_clean/*.bmp')
#origin="/home/kali/.keras/datasets/DB_augmented/"
origin="/home/kali/Bureau/PFE/newGit/AuthentificationBiometrique/database/"
images = glob.glob('{}*.bmp'.format(origin))

directories=[]
x=9 # longueur chemin avant nom image # or use split
for img in images: 
	os.system('python3 /home/kali/Bureau/PFE/newGit/AuthentificationBiometrique/pycode_traitement_images/traitement_imagesv1.py -i {} -o {}'.format(img,img))
	short=img.split("/")
	short=short[-1]
	#print(short)
	#print(short[:-6])
	directory=short[:-6]
	d=origin+directory
	if d not in directories:
		os.system("mkdir {}".format(d))
		directories.append(d)
	os.system("mv {} {}".format(img,d))

