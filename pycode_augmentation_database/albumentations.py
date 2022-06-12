import albumentations as A
import matplotlib.pyplot as plt
import argparse
import cv2

def visualize(image):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image)

parser = argparse.ArgumentParser()

parser.add_argument('-i','--image',
required=True,
dest='image',
help='select image of the finger'
)

args = parser.parse_args()

image = cv2.imread(args.image,0)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

transform = A.Compose(
    [A.CLAHE(),
     A.RandomRotate90(),
     A.Transpose(),
     A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50,
                        rotate_limit=45, p=.75),
     A.Blur(blur_limit=3),
     A.OpticalDistortion(),
     A.GridDistortion(),
     A.HueSaturationValue()])

augmented_image = transform(image=image)['image']
visualize(augmented_image)