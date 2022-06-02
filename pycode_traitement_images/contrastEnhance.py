import argparse
from PIL import Image, ImageFilter
from PIL import ImageEnhance

parser = argparse.ArgumentParser()

parser.add_argument('-i','--image',
required=True,
dest='image',
help='select image of the finger'
)


args = parser.parse_args()


enhanceFactor=2.2
#im = Image.open( 'img.jpg' )
im = Image.open( args.image )

lenX=im.size[0]
lenY=im.size[1]
tabOrg=[]

for x in range(0,lenX):
	for y in range(0,lenY):
		coo=(x,y)
		#print(im.getpixel(coo))
		tabOrg.append(im.getpixel(coo))

im.show()

enh = ImageEnhance.Contrast(im)
enh.enhance(enhanceFactor).show("improved more contrast")

enh.enhance(enhanceFactor).save("contrasted3.bmp")

tabFin=[]

for x in range(0,lenX):
	for y in range(0,lenY):
		coor=(x,y)
		#print(enh.enhance(enhanceFactor).getpixel(coor))
		tabFin.append(im.getpixel(coor))


def tabs():
	print("--tabs--")
	print(tabOrg)
	print("---")
	print(tabFin)


	diff=[]
	print(len(tabOrg))
	for e in range(0,len(tabOrg)):
		diff.append(tabFin[e][0]-tabOrg[e][0])

	print(diff)

#tabs()



