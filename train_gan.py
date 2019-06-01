import matplotlib.pyplot as plt
from random import choice, sample
import os
import numpy as np
import argparse

from keras.preprocessing.image import img_to_array, array_to_img, load_img

from keras.layers import UpSampling2D, Conv2D, BatchNormalization, Activation
from keras.layers import Input, Dense, Reshape, LeakyReLU, Dropout, ZeroPadding2D, Flatten

from keras.models import Model, Sequential

from keras.optimizers import Adam

IMG_WIDTH = 128
IMG_HEIGHT = 128
CHANNELS = 3

def getRealMiniBatch(path='allPhotos',batch_size=32):
  allp = os.listdir(path)
  while True:
    batch = sample(allp,batch_size)
    imgs = []
    for img in batch:
      x = os.path.join(path,img)
      x = img_to_array(load_img( x , target_size=(IMG_WIDTH, IMG_HEIGHT) ))
      imgs.append(x)
    yield (np.array(imgs)/127.5)-1

def Generator(INP_DIM=100):
    def Block(inp, N, kernel_size=3):
      x = UpSampling2D()(inp)
      x = Conv2D(N, kernel_size=kernel_size, padding='same')(x)
      x = BatchNormalization(momentum=0.8)(x)
      x = Activation('relu')(x)
      return x
    
    inp = Input( shape=(INP_DIM,) )
    x = Dense( 512*4*4, activation='relu' )(inp)
    x = Reshape( (4,4,512) )(x)

    x = Block(x, 512)
    x = Block(x, 512)
    x = Block(x, 256)
    x = Block(x, 256)
    x = Block(x, 128)

    x = Conv2D( 3, kernel_size=3, padding='same' )(x)
    out = Activation('tanh')(x)

    generator = Model(inp, out)
    return generator
  
def Descriminator(width, height, channels):
    def Block(inp, N, kernel_size = 3, strides=2,zeropad=False):
      x = Dropout(0.3)(inp)
      x = Conv2D( N, kernel_size=kernel_size, strides=strides, padding='same' )(x)
      if zeropad:
        x = ZeroPadding2D(padding=((0,1),(0,1)))(x)
      x = BatchNormalization(momentum=0.8)(x)
      x = LeakyReLU(alpha=0.2)(x)
      return x

    inp = Input( shape=(width,height,channels) )
    x = Conv2D( 32, kernel_size=3, strides=2, padding='same' )(inp)
    x = LeakyReLU(alpha=0.2)(x)

    x = Block(x, 64, zeropad=True)
    x = Block(x, 128)
    x = Block(x, 256, strides=1)
    x = Block(x, 512, strides=1)

    x = Dropout(0.25)(x)
    x = Flatten()(x)
    out = Dense(1, activation='sigmoid')(x)

    descriminator = Model(inp,out)
    return descriminator
  
  
def GAN(descriminator,generator,INP_DIM=100):
  descriminator.trainable=False
  inp = Input( shape=(INP_DIM,) )
  x = generator(inp)
  out = descriminator(x)
  return Model(inp,out)

#Save 25 generated images for demonstration purposes using matplotlib.pyplot.
def save_figure(epoch,rows,columns,INP_DIM=100):
    noise = np.random.normal(0, 1, (rows * columns, INP_DIM))
    generated_images = generator.predict(noise)
    
    generated_images = generated_images/2 + 0.5
    
    figure, axis = plt.subplots(rows, columns)
    image_count = 0
    for row in range(rows):
        for column in range(columns):
            axis[row,column].imshow(generated_images[image_count, :], cmap='spring')
            axis[row,column].axis('off')
            image_count += 1
    figure.savefig("generated_images/generated_%d.png" % epoch)
    plt.close()

def getFakeMiniBatch(batch_size=32, INP_DIM=100):
  while True:
    x = np.random.normal(np.random.normal(0,1,(32,INP_DIM)))
    yield generator.predict(x),x

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
  parser.add_argument("-w", "--weights", default="saved_weights",help="Model weights directory")
	parser.add_argument("-p", "--photos_dir", default="allPhotos",help="Faces directory")
	parser.add_argument("-b", "--batch_size", type=int, default=32, help="Size of each batch")
	parser.add_argument("-i", "--inp_dim", type=int, default=100, help="Size of Input Noise")
	parser.add_argument("-e", "--epochs", type=int, default=10000, help="Number of Epochs to train")

	args = parser.parse_args()

	BATCH_SIZE = args.batch_size
	INP_DIM = args.inp_dim
	EPOCHS = args.epochs

	optimizer = Adam(0.0002,0.5)

	descriminator = Descriminator(width=IMG_WIDTH, height=IMG_HEIGHT, channels=CHANNELS)
	descriminator.compile( loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'] )

	generator = Generator( INP_DIM=INP_DIM )
	
	gan = GAN(descriminator,generator,INP_DIM=INP_DIM)
	gan.compile( loss="binary_crossentropy", optimizer=optimizer )

	if os.path.exists(args.weights):
	  generator.load_weights(args.weights+'/generator_weights.h5')
	  descriminator.load_weights(args.weights+'/descriminator_weights.h5')
	  gan.load_weights(args.weights+'/gan_weights.h5')
	  print('Using Pretrained Networks...')
	else:
	  os.mkdir(args.weights)

	real_labels = np.ones((BATCH_SIZE,1))
	fake_labels = np.zeros((BATCH_SIZE,1))
	real_image_gen = getRealMiniBatch(path=args.photos_dir, batch_size=BATCH_SIZE)
	fake_image_gen = getFakeMiniBatch(batch_size=BATCH_SIZE, INP_DIM=INP_DIM )

	for epoch in range(1,EPOCHS+1):
	  fake_images,noise = next(fake_image_gen)
	  real_images = next(real_image_gen)
	  
	  descri_loss_real = descriminator.train_on_batch( real_images, real_labels )
	  descri_loss_fake = descriminator.train_on_batch( fake_images, fake_labels )
	  descriminator_loss = 0.5*np.add(descri_loss_real,descri_loss_fake)
	  
	  gan_loss = gan.train_on_batch( noise, real_labels)
	  
	  history.append( (descriminator_loss, gan_loss) )
	  if epoch % 250 ==0:
	    print( "{} [ Descriminator loss : {:.2f} acc : {:.2f} ] [ Generator loss : {:.2f} ]".format( 
	        epoch, descriminator_loss[0], 100*descriminator_loss[1], gan_loss ) )
	    
	  if epoch % 500 == 0:
	    if not os.path.exists('generated_images'): os.mkdir('generated_images')
	    descriminator.save_weights(args.weights+'/descriminator_weights.h5')
	    generator.save_weights(args.weights+'/generator_weights.h5')
	    gan.save_weights(args.weights+'/gan_weights.h5')
	    save_figure(epoch,5,5,INP_DIM)
	    print("Figure Saved | All Weights Saved")