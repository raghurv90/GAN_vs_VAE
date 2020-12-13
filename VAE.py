import keras
from keras import layers
from keras.datasets.mnist import load_data
import numpy as np
from matplotlib import pyplot
from keras import backend as K

imageDim = 28 * 28
hiddenDim = 256
# intermediate_dim2 = 64
latentDim = 2

def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latentDim), mean=0., stddev=0.1)
    return z_mean + K.exp(z_log_sigma) * epsilon

def getLatentPoints(latentDim, numSamples=1000):
	return randn(latentDim * numSamples).reshape(numSamples, latentDim)

def evaluateLoss(inputs, outputs, z_log_sigma, z_mean)
	reconstruction_loss = imageDim*(keras.losses.binary_crossentropy(inputs, outputs))
	temp = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
	kl_loss = -0.5*(K.sum(temp, axis=-1))
	return K.mean(reconstruction_loss + kl_loss)

def printImages(n=6) :
  decoded_imgs = decoder.predict(getLatentPoints(latentDim))
  for i in range(n**2):
    pyplot.subplot(n, n, 1 + i)
    pyplot.axis('off')
    pyplot.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray_r')
  pyplot.show()

(x_train, _), (x_test, _) = load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# Create encoder
inputs = keras.Input(shape=(imageDim,))
h = layers.Dense(hiddenDim, activation='sigmoid')(inputs)
# h = layers.Dense(intermediate_dim2, activation='sigmoid')(h)

z_mean = layers.Dense(latentDim)(h)
z_log_sigma = layers.Dense(latentDim)(h)
z = layers.Lambda(sampling)([z_mean, z_log_sigma])

encoder = keras.Model(inputs, [z_mean, z_log_sigma, z], name='encoder')

# Create decoder
latent_inputs = keras.Input(shape=(latentDim,), name='z_sampling')
x = layers.Dense(hiddenDim, activation='sigmoid')(latent_inputs)
outputs = layers.Dense(imageDim, activation='sigmoid')(x)
decoder = keras.Model(latent_inputs, outputs, name='decoder')

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = keras.Model(inputs, outputs, name='vae')
vae.add_loss(evaluateLoss(inputs, outputs, z_log_sigma, z_mean))
vae.compile(optimizer='adam')
vae.fit(x_train, x_train,epochs=100, batch_size=128, validation_data=(x_test, x_test))

printImages()


