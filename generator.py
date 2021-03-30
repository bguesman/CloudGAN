import tensorflow as tf

"""

This class is where we define the discriminator as an object.

"""

class Generator(tf.keras.Model):
    def __init__(self, learning_rate=0.00002):

        # I don't know why this is here, but I'm afraid to touch it.
        ######vvv DO NOT CHANGE vvvv##############
        super(Generator, self).__init__()
        ######^^^ DO NOT CHANGE ^^^##################

        # Define the optimizer we want to use to train the model.
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)

        # Dimension of the latent space.
        self.latent_dimension = 100

        # First use some dense layers to transform latent space
        self.d0 = tf.keras.layers.Dense(128, use_bias=True)
        self.b0 = tf.keras.layers.BatchNormalization()
        self.lr0 = tf.keras.layers.LeakyReLU()
        # -- Start at 16 x 16 --
        self.d1 = tf.keras.layers.Dense(256, use_bias=True)

        self.lr2 = tf.keras.layers.LeakyReLU()
        self.c2 = tf.keras.layers.Conv2D(filters=16, kernel_size=5, strides=1, padding='same')
        self.b2 = tf.keras.layers.BatchNormalization()
        # -- Scale to 64x64 --
        self.lr3 = tf.keras.layers.LeakyReLU()
        self.c3 = tf.keras.layers.Conv2D(filters=16, kernel_size=5, strides=1, padding='same')
        self.b3 = tf.keras.layers.BatchNormalization()
        # -- Scale to 128x128 --
        self.lr4 = tf.keras.layers.LeakyReLU()
        self.c4 = tf.keras.layers.Conv2D(filters=8, kernel_size=5, strides=1, padding='same')
        self.b4 = tf.keras.layers.BatchNormalization()
        # -- Scale to 256x256 --
        self.c5 = tf.keras.layers.Conv2D(filters=1, kernel_size=5, strides=1, padding='same', use_bias=False)

    @tf.function
    def call(self, input):
        # Our input is an element of the latent space, so it has dimension
        # self.latent_dimension. We'll apply a dense layer and reshape to a 16x16 grid.
        grid = tf.reshape(self.d1(self.lr0(self.b0(self.d0(input)))), (input.shape[0], 16, 16, 1))

        # Next we'll apply a series of convolutional layers and batch norms, 
        # "decoding" the latent state.
        conv2 = self.lr2(self.b2(self.c2(grid)))
        # -- Scale to 64x64 --
        conv2 = tf.image.resize(conv2, [64, 64])
        conv3 = self.lr3(self.b3(self.c3(conv2)))
        # -- Scale to 128x128 --
        conv3 = tf.image.resize(conv3, [128, 128])
        conv4 = self.lr4(self.b4(self.c4(conv3)))
        # -- Scale to 256x256 --
        conv4 = tf.image.resize(conv4, [256, 256])
        conv5 = self.c5(conv4)
        
        return conv5
