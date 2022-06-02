import numpy as np
import cv2 
import sys
from matplotlib import pyplot as plt

def effacementcontour(edges, cols, rows):
	img = edges
	for i in range(rows):
		maxj = 0
		minj = cols
		for j in range(int(cols/2), 0, -1):
			#trouver le pixxel plus proche de milieu
			if(img[i,j] != 0 and maxj < j) :
				maxj = j
			else :
				# mettre les autres pixels de la ligne à 0
				if(img[i,j] != 0) :
					img[i,j] = 0
					
		for j in range(int(cols/2), cols):
			if(img[i,j] != 0 and minj > j) :
				minj = j
			else :
				# mettre les autres pixels de la ligne à 0
				if(img[i,j] != 0) :
					img[i,j] = 0
	return img
			
def rognerImage(img, edges) :
	rows, cols = edges.shape
	minCol = 0
	maxCol = cols

	for i in range(rows):
		for j in range(int(cols/2), 0, -1):
			if(edges[i,j] != 0 and minCol < j) :
				print(j)
				minCol = j
		for j in range(int(cols/2), cols):
			if(edges[i,j] != 0 and maxCol > j) :
				print(j)
				maxCol = j	
	print(minCol, maxCol)	
	#Rognage		
	cropped = img[0:rows, minCol:maxCol]	
	return cropped

imagearg = sys.argv[1]
name = imagearg.split('.')[0]	
img = cv2.imread(imagearg,0)
img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
edges = cv2.Canny(img,100,200)

rows, cols = edges.shape

edgesmodif1 = effacementcontour(edges, cols, rows)
edgesmodif2 = rognerImage(img, edgesmodif1)
#write image modified
cv2.imwrite(name + "modified.bmp", edgesmodif2)

plt.subplot(211),plt.imshow(edgesmodif1,cmap = 'gray')
plt.title('Edge Detection using Canny after modif1'), plt.xticks([]), plt.yticks([])
plt.subplot(212),plt.imshow(edgesmodif2,cmap = 'gray')
plt.title('Edge Detection using Canny after modif 2'), plt.xticks([]), plt.yticks([])
plt.show()



				
