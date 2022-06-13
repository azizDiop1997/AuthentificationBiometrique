import argparse
import cv2 
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
				minCol = j
		for j in range(int(cols/2), cols):
			if(edges[i,j] != 0 and maxCol > j) :
				maxCol = j		
	#Rognage		
	cropped = img[0:rows, minCol:maxCol]	
	return cropped

def rognertrntpercent(img) :
    rows, cols = img.shape
    imageR = img[0:int(rows/3), 0:cols]
    return imageR

def resize(img) :
	dim = (120, 320)
	# resize image
	resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	return resized
	
parser = argparse.ArgumentParser()

parser.add_argument('-i','--image',
required=True,
dest='image',
help='select image of the finger'
)

parser.add_argument('-o','--output',
required=True,
dest='outputImg',
help='name of the (tmp) output image'
)

parser.add_argument('-v','--verbose',
required=False,
dest='display',
help='afficher l\'image'
)


args = parser.parse_args()

#print(args.image)
img = cv2.imread(args.image,0)
img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
edges = cv2.Canny(img,100,200)

rows, cols = edges.shape

edgesmodif1 = effacementcontour(edges, cols, rows)
edgesmodif2 = rognerImage(img, edgesmodif1)
edgesresize = resize(edgesmodif2)
#edgesmodif3 = rognertrntpercent(edgesmodif2)
#write image modified
cv2.imwrite(args.outputImg, edgesresize) # "testmodified.bmp"

if(args.display) :
	plt.subplot(211),plt.imshow(edgesmodif2,cmap = 'gray')
	plt.title('Edge Detection using Canny after modif1'), plt.xticks([]), plt.yticks([])
	plt.subplot(212),plt.imshow(edgesresize,cmap = 'gray')
	plt.title('Edge Detection using Canny after modif 2'), plt.xticks([]), plt.yticks([])
	plt.show()



				

