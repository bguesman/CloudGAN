import tensorflow as tf

"""

This class is where we define the discriminator as an object.

"""

class Generator(tf.keras.Model):
    def __init__(self, learning_rate=0.0005):

        # I don't know why this is here, but I'm afraid to touch it.
        ######vvv DO NOT CHANGE vvvv##############
        super(Generator, self).__init__()
        ######^^^ DO NOT CHANGE ^^^##################

        # Define the optimizer we want to use to train the model.
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)

        # Dimension of the latent space.
        self.latent_dimension = 128

        # No activation for the first layer, which is mostly just there
        # to aid the reshape.
        self.d1 = tf.keras.layers.Dense(64)
        self.b0 = tf.keras.layers.BatchNormalization()

        # Goes to 16x16
        self.lr = tf.keras.layers.LeakyReLU();
        self.c2 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=5, strides=2, padding='same')
        self.b2 = tf.keras.layers.BatchNormalization()
        # Goes to 32x32
        self.c3 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=5, strides=2, padding='same')
        self.b3 = tf.keras.layers.BatchNormalization()
        # Goes to 64x64
        self.c4 = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=5, strides=2, padding='same')
        self.b4 = tf.keras.layers.BatchNormalization()
        # Goes to 128x128
        self.c5 = tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=5, strides=2, padding='same')
        self.b5 = tf.keras.layers.BatchNormalization()
        # Goes to 256x256 (final layers)
        self.c6 = tf.keras.layers.Conv2DTranspose(filters=4, kernel_size=5, strides=2, padding='same')
        self.b6 = tf.keras.layers.BatchNormalization()
        self.c7 = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=5, strides=1, padding='same')

    @tf.function
    def call(self, input):
        # Our input is an element of the latent space, so it has dimension
        # self.latent_dimension. We'll apply a dense layer and reshape to a grid.
        grid = tf.reshape(self.lr(self.b0(self.d1(input))), (input.shape[0], 8, 8, 1))
        # grid = tf.reshape(self.d1(input), (input.shape[0], 256, 256, 1))

        # Next we'll apply a series of convolutional layers and batch norms, 
        # "decoding" the latent state.
        conv2 = self.lr(self.b2(self.c2(grid)))
        conv3 = self.lr(self.b3(self.c3(conv2)))
        conv4 = self.lr(self.b4(self.c4(conv3)))
        conv5 = self.lr(self.b5(self.c5(conv4)))
        conv6 = self.lr(self.b6(self.c6(conv5)))
        conv7 = self.c7(conv6)
        
        return conv7
