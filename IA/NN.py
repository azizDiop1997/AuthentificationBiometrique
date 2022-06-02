import numpy as np
import matplotlib.pyplot as plt
from PIL import Image#, ImageFilter

class NeuralNetwork:
    def __init__(self, learning_rate,nbPixels):
        print(type(np.random.randn()))
        #self.weights = np.array([np.random.randn(), np.random.randn()])
        #print(type(self.weights))
        tmpArr = []
        print(type(tmpArr))

        for i in range(nbPixels):
                tmpArr.append(np.random.randn())     
        self.weights=np.array(tmpArr) # list into nparray
# of course your final object takes twice the space in the memory at the creation step, but appending on python list is very fast, and creation using np.array() also.
        print(type(self.weights))
        #print(self.weights)
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
        print("******")
        #print(input_vector)
        #print(target)
        #print(self.weights)
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2
        print(layer_1)
        derror_dprediction = 2 * (prediction - target)
        dprediction_dlayer1 = self._sigmoid_deriv(layer_1)
        dlayer1_dbias = 1
        dlayer1_dweights = (0 * self.weights) + (1 * input_vector)
        print("p")
        derror_dbias = (derror_dprediction * dprediction_dlayer1 * dlayer1_dbias)
        print("o")
        print(derror_dprediction)
        print(dprediction_dlayer1)
        print(type(dlayer1_dweights))
        derror_dweights = (derror_dprediction * dprediction_dlayer1 * dlayer1_dweights)
        print("d")
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
trainingSet = []

im = Image.open( "../pycode_traitement_images/contrasted.bmp" )

#lenX=im.size[0]
lenX=7
#lenY=im.size[1] 
lenY=7
print(lenX,lenY)
tab=[]

for x in range(0,lenX):
	for y in range(0,lenY):
		coo=(x,y)
		#print(im.getpixel(coo))
		tab.append(im.getpixel(coo)/255) # (x,y)
print(len(tab))
im.show()

# images of the trainset
for i in range(8):
	trainingSet.append(tab)
	#print(i)
#################################



learning_rate = 0.1


input_vectors=trainingSet

input_vectors2 = np.array(
    [
        [3, 1.5],
        [2, 1],
        [4, 1.5],
        [3, 4],
        [3.5, 0.5],
        [2, 0.5],
        [5.5, 1],
        [1, 1],
    ]
)

#print(input_vectors)
#print(input_vectors2)

targets = np.array([0, 1, 0, 1, 0, 1, 1, 0])

learning_rate = 0.1

nbPixels=49 # fixed size for all images ?
neural_network = NeuralNetwork(learning_rate,nbPixels)

training_error = neural_network.train(input_vectors, targets, 10000)

plt.plot(training_error)
plt.xlabel("Iterations")
plt.ylabel("Error for all training instances")
plt.savefig("cumulative_error.png")
plt.show()




