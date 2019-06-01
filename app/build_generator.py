from keras.models import Model
from keras.layers import Input, Dense, Reshape
from keras.layers import UpSampling2D, Conv2D, BatchNormalization, Activation

def Generator(weights):
    def Block(inp, N, kernel_size=3):
      x = UpSampling2D()(inp)
      x = Conv2D(N, kernel_size=kernel_size, padding='same')(x)
      x = BatchNormalization(momentum=0.8)(x)
      x = Activation('relu')(x)
      return x
    
    inp = Input( shape=(100,) )
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
    generator.load_weights(weights)
    return generator