from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
from numpy.random import randn
from numpy.random import randint
from keras.datasets.mnist import load_data
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from matplotlib import pyplot

def getRealData(dataset, numSamples):
	return dataset[randint(0, dataset.shape[0], numSamples)], ones((numSamples, 1))
	 
def getLatentPoints(latentDim, numSamples):
	return randn(latentDim * numSamples).reshape(numSamples, latentDim)

def getNoiseData(genModel, latentDim, numSamples):
	return genModel.predict(getLatentPoints(latentDim, numSamples)), zeros((numSamples, 1))

def printImages(n = 6) :
  x_input = getLatentPoints(100, 100)
  trainX = genModel.predict(x_input)
  print(genModel.summary(line_length=None, positions=None, print_fn=None))
  print(disModel.summary(line_length=None, positions=None, print_fn=None))
  for i in range(n**2):
    pyplot.subplot(n, n, 1 + i)
    pyplot.axis('off')
    pyplot.imshow(trainX[i, :, :, 0], cmap='gray_r')
  pyplot.show()

def saveImageFile(examples, epoch, n=6):
	for i in range(n**2):
		pyplot.subplot(n, n, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
	f = 'generated_plot_e%03d.png' % (epoch+1)
	pyplot.savefig(f)
	pyplot.close()

def modelDiscriminator(inShape=(28,28,1)):
	model = Sequential()
	model.add(Conv2D(128, 3, strides=2, padding='same', input_shape=inShape))
	model.add(LeakyReLU(alpha=0.2))
	# model.add(Dropout(0.4))
	model.add(Conv2D(128, 3, strides=2, padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# model.add(Dropout(0.4))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

def modelGenerator(latentDim, numNodes = 128 * 7 * 7):
	model = Sequential()
	model.add(Dense(numNodes, input_dim=latentDim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((7, 7, 128)))
	#upsampling to 14x14
	model.add(Conv2DTranspose(128, 4, strides=2, padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	#upsampling to 28x28
	model.add(Conv2DTranspose(128, 4, strides=2, padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))
	return model

def modelGAN(genModel, disModel):
	disModel.trainable = False
	model = Sequential()
	model.add(genModel)
	model.add(disModel)
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model

def summarize(epoch, genModel, disModel, dataset, latentDim, numSamples=100):
	x1,y1 = getRealData(dataset, numSamples)
	temp, a1 = disModel.evaluate(x1, y1, verbose=0)
	x2, y2 = getNoiseData(genModel, latentDim, numSamples)
	temp, a2 = disModel.evaluate(x2, y2, verbose=0)
	print('>Real Accuracy is given by : %.0f%%, and fake accuracy is given by : %.0f%%' % (a1*100, a2*100))
	saveImageFile(x2, epoch)
	f = 'generator_model_%03d.h5' % (epoch + 1)
	genModel.save(f)

def train(genModel, disModel, gan, dataset, latentDim, numEpochs=100, batchSize=128):
	numBatches = int(dataset.shape[0] / batchSize)
	dataSize = int(batchSize / 2)
	for i in range(numEpochs):
		for j in range(numBatches):
			x1, y1 = getRealData(dataset, dataSize)
			x2, y2 = getNoiseData(genModel, latentDim, dataSize)
			x, y = vstack((x1, x2)), vstack((y1, y2))
			#train on real samples
			d_loss, temp = disModel.train_on_batch(x, y)
			x_gan, y_gan = getLatentPoints(latentDim, batchSize), ones((batchSize, 1))
			#train on fake samples
			g_loss = gan.train_on_batch(x_gan, y_gan)
		print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, numBatches, d_loss, g_loss))
		if (i+1) % 5 == 0:
			summarize(i, genModel, disModel, dataset, latentDim)

latentDim = 100
disModel = modelDiscriminatorModel()
genModel = modelGeneratorModel(latentDim)
gan = modelGAN(genModel, disModel)
(trainx, temp1), (temp2, temp3) = load_data()
dataset = expand_dims(trainx, axis=-1).astype('float32')/255.0
train(genModel, disModel, gan, dataset, latentDim)

printImages()
