import numpy as np
import matplotlib.pyplot as plt
from PIL import Image#, ImageFilter
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-i','--image',
required=True,
dest='image',
help='select image of the finger'
)

parser.add_argument('-t','--train',
dest='train',
help='add this option to train the neural network'
)

args = parser.parse_args()


learning_rate = 0.1
#nbPixels=0

class NeuralNetwork:
    def __init__(self, learning_rate,nbPixels):
        #print(type(np.random.randn()))
        #self.weights = np.array([np.random.randn(), np.random.randn()])
        #print(type(self.weights))
        tmpArr = []

        for i in range(nbPixels):
                tmpArr.append(np.random.randn())
        self.weights=np.array(tmpArr) # list into nparray
# of course your final object takes twice the space in the memory at the creation step, but appending on python list is very fast, and creation using np.array() also.

        #print(type(self.weights))
        print(len(self.weights))
        self.bias = np.random.randn()
        self.learning_rate = learning_rate
        #print(self.bias)
        #print(self.learning_rate)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_deriv(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def predict(self, input_vector):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2
        #print("predicts:",prediction)
        return prediction

    def _compute_gradients(self, input_vector, target):
        #print("******")
        #print(input_vector)
        #print(target)
        #print(self.weights)
        #print("*",len(input_vector))
        #print(len(self.weights))
	
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2
        #print(layer_1)
        derror_dprediction = 2 * (prediction - target)
        dprediction_dlayer1 = self._sigmoid_deriv(layer_1)
        dlayer1_dbias = 1
        dlayer1_dweights = (0 * self.weights) + (1 * input_vector)
        derror_dbias = (derror_dprediction * dprediction_dlayer1 * dlayer1_dbias)
        #print(derror_dprediction)
        #print(dprediction_dlayer1)
        #print(type(dlayer1_dweights))
        derror_dweights = (derror_dprediction * dprediction_dlayer1 * dlayer1_dweights)
        return derror_dbias, derror_dweights

    def _update_parameters(self, derror_dbias, derror_dweights):
        self.bias = self.bias - (derror_dbias * self.learning_rate)
        self.weights = self.weights - (
            derror_dweights * self.learning_rate
        )

    def train(self, input_vectors, targets, iterations):
        cumulative_errors = []
        for current_iteration in range(iterations):
            # Pick a data instance at random
            random_data_index = np.random.randint(len(input_vectors))

            input_vector = input_vectors[random_data_index]
            target = targets[random_data_index]

            # Compute the gradients and update the weights
            derror_dbias, derror_dweights = self._compute_gradients(
                input_vector, target
            )

            self._update_parameters(derror_dbias, derror_dweights)

            # Measure the cumulative error for all the instances
            if current_iteration % 100 == 0:
                cumulative_error = 0
                # Loop through all the instances to measure the error
                for data_instance_index in range(len(input_vectors)):
                    data_point = input_vectors[data_instance_index]
                    target = targets[data_instance_index]

                    prediction = self.predict(data_point)
                    error = np.square(prediction - target)

                    cumulative_error = cumulative_error + error
                cumulative_errors.append(cumulative_error)

        return cumulative_errors



#################################
def train():
	trainingSet = []
	imagesTraining = ["11.bmp","12.bmp","13.bmp","21.bmp","22.bmp","23.bmp"]# images preprocessed by our algorithms
	lenImg=[]
	print("images of the training set:")
	for image in imagesTraining:
		print(image)
		#im = Image.open( args.image )
		im = Image.open(image)
		lenX=im.size[0]
		lenY=im.size[1]
		tab=[]
		for x in range(0,lenX):
			for y in range(0,lenY):
				coo=(x,y)
				#print(im.getpixel(coo))
				tab.append(im.getpixel(coo)/255) # (x,y)
		print("len: ",len(tab))
		lenImg.append(len(tab))
		#im.show()
		trainingSet.append(tab)
		##
	print(lenImg)
	nbPixels=min(lenImg) # here we take th1e smallest image in order to train the NN on fixed sizes images
	nbPixels=32320
	print("nbPixels:",nbPixels)
# training set doit contenir 1 image de chaque personne (ça représente l'image d'enrollement)

	vectors=[]
	for l in trainingSet:
		print("size:",len(l[:nbPixels]))
		vectors.append(l[:nbPixels]) # all vectors are now of the same size
#	input_vectors=trainingSet
	print(len(vectors))
	print(len(vectors[0]))

	input_vectors=np.array(vectors)

	targets = np.array([0, 0, 0, 1, 1, 1])

	neural_network = NeuralNetwork(learning_rate,nbPixels)

	training_error = neural_network.train(input_vectors, targets, 10000)


	plt.plot(training_error)
	plt.xlabel("Iterations")
	plt.ylabel("Error for all training instances")
	plt.savefig("cumulative_error.png")
	plt.show()
#################################



# tester une image, voir si personne reconnue
def test():
	im = Image.open(args.image)
	lenX=im.size[0]
	lenY=im.size[1]
	tab=[]
	for x in range(0,lenX):
		for y in range(0,lenY):
			coo=(x,y)
			tab.append(im.getpixel(coo)/255)

	#nbP=34880
	nbP=32320
	
	neural_network = NeuralNetwork(learning_rate,nbP)
	tab2=tab[:nbP]
	print("pred:",neural_network.predict(np.array(tab2)))

	# fixed size for all images ?
	# prendre en arg x*y de l'image passée en paramètre est + simple mais est ce que le ML peut apprendre 
	# efficacement avec des images de tailles différentes


## main
if train:
	print("neural network started training...")
	train()
	print("training terminated")

print("dd")
test()

