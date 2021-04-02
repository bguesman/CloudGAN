import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

import tensorflow_addons as tfa

"""

This class is where we define the generator as an object.

"""

class Generator(tf.keras.Model):
    def __init__(self, learning_rate=1e-5):

        # I don't know why this is here, but I'm afraid to touch it.
        ######vvv DO NOT CHANGE vvvv##############
        super(Generator, self).__init__()
        ######^^^ DO NOT CHANGE ^^^##################

        # Define the optimizer we want to use to train the model.
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)

        # Dimension of the latent space.
        self.latent_dimension = 100

        self.initialize_std_dev = 0.1
        
        # First use some dense layers to transform latent space
        self.d0 = tf.keras.layers.Dense(64, use_bias=True, kernel_initializer=tf.keras.initializers.TruncatedNormal(0, self.initialize_std_dev), bias_initializer=tf.keras.initializers.TruncatedNormal(0, self.initialize_std_dev))
        self.b0 = tf.keras.layers.BatchNormalization()
        self.lr0 = tf.keras.layers.LeakyReLU()
        # -- Start at 4 x 4 x 512 channels --
        self.d1 = tf.keras.layers.Dense(4*4*128, use_bias=False, kernel_initializer=tf.keras.initializers.TruncatedNormal(0, self.initialize_std_dev), bias_initializer=tf.keras.initializers.TruncatedNormal(0, self.initialize_std_dev))
        
        # Batch norms and activations for convolutional layers.
        self.lr = [tf.keras.layers.LeakyReLU() for i in range(7)]
        self.b = [tf.keras.layers.BatchNormalization() for i in range(7)]

        # 4x4 => 256x256 means we need 6 convolutional layers, plus one final one
        self.c0 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', kernel_initializer=tf.keras.initializers.TruncatedNormal(0, self.initialize_std_dev), bias_initializer=tf.keras.initializers.TruncatedNormal(0, self.initialize_std_dev))
        self.c1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', kernel_initializer=tf.keras.initializers.TruncatedNormal(0, self.initialize_std_dev), bias_initializer=tf.keras.initializers.TruncatedNormal(0, self.initialize_std_dev))
        self.c2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', kernel_initializer=tf.keras.initializers.TruncatedNormal(0, self.initialize_std_dev), bias_initializer=tf.keras.initializers.TruncatedNormal(0, self.initialize_std_dev))
        self.c3 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', kernel_initializer=tf.keras.initializers.TruncatedNormal(0, self.initialize_std_dev), bias_initializer=tf.keras.initializers.TruncatedNormal(0, self.initialize_std_dev))
        self.c4 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', kernel_initializer=tf.keras.initializers.TruncatedNormal(0, self.initialize_std_dev), bias_initializer=tf.keras.initializers.TruncatedNormal(0, self.initialize_std_dev))
        self.c5 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', kernel_initializer=tf.keras.initializers.TruncatedNormal(0, self.initialize_std_dev), bias_initializer=tf.keras.initializers.TruncatedNormal(0, self.initialize_std_dev))
        self.c6 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', kernel_initializer=tf.keras.initializers.TruncatedNormal(0, self.initialize_std_dev), bias_initializer=tf.keras.initializers.TruncatedNormal(0, self.initialize_std_dev))
        # Final Convolution
        self.c7 = tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=1, padding='same', kernel_initializer=tf.keras.initializers.TruncatedNormal(0, self.initialize_std_dev), bias_initializer=tf.keras.initializers.TruncatedNormal(0, self.initialize_std_dev))

    @tf.function
    def call(self, input):
        # Our input is an element of the latent space, so it has dimension
        # self.latent_dimension. We'll apply a dense layer and reshape to a 16x16 grid.
        grid = tf.reshape(self.d1(self.lr0(self.b0(self.d0(input)))), (input.shape[0], 4, 4, 128))

        # Pipe through convolutions/upscales.
        conv0 = tf.image.resize(self.lr[0](self.b[0](self.c0(grid))), [8, 8])
        conv1 = tf.image.resize(self.lr[1](self.b[1](self.c1(conv0))), [16, 16])
        conv2 = tf.image.resize(self.lr[2](self.b[2](self.c2(conv1))), [32, 32])
        conv3 = tf.image.resize(self.lr[3](self.b[3](self.c3(conv2))), [64, 64])
        conv4 = tf.image.resize(self.lr[4](self.b[4](self.c4(conv3))), [128, 128])
        conv5 = tf.image.resize(self.lr[5](self.b[5](self.c5(conv4))), [256, 256])
        conv6 = self.lr[6](self.b[6](self.c6(conv5)))

        # Final convolution
        return self.c7(conv6)
