import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

import tensorflow_addons as tfa

"""

This class is where we define the generator as an object.

"""

class StyleGenerator(tf.keras.Model):
    def __init__(self, learning_rate=2e-5):

        # I don't know why this is here, but I'm afraid to touch it.
        ######vvv DO NOT CHANGE vvvv##############
        super(StyleGenerator, self).__init__()
        ######^^^ DO NOT CHANGE ^^^##################

        # Define the optimizer we want to use to train the model.
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)

        # Random seed to control randomness of model initialization.
        self.seed = 1

        # Dimension of the latent space.
        self.latent_dimension = 100
        # Standard deviation of normal distribution used to initialize weights
        self.initialize_std_dev = 0.05

        # The generated image begins as a random constant 4x4 image with 512 channels.
        self.synthesis_base = tf.Variable(tf.random.normal([1, 4, 4, 64], seed=self.seed))

        # The style network takes a latent vector and pipes it through an 8-layer MLP to
        # generate a "style" code.
        self.map_depth = 8
        self.map_dimension = 512
        self.map_dense = [tf.keras.layers.Dense(self.map_dimension, use_bias=True, 
            kernel_initializer=tf.keras.initializers.TruncatedNormal(0, self.initialize_std_dev), 
            bias_initializer=tf.keras.initializers.TruncatedNormal(0, self.initialize_std_dev)) 
            for _ in range(self.map_depth)]
        self.map_activation = [tf.keras.layers.LeakyReLU() for _ in range(self.map_depth)]

        # The synthesis network takes the synthesis_base and upscales it to an image, injecting
        # the style vector and some noise along the way.
        self.convs_per_resolution = 1
        self.resolutions = [4, 8, 16, 32, 64, 128, 256]
        self.features = [16, 16, 8, 8, 8, 4, 1]
        self.synthesis_conv = [[tf.keras.layers.Conv2D(filters=feature_count, kernel_size=3, 
            strides=1, padding='same', 
            kernel_initializer=tf.keras.initializers.TruncatedNormal(0, self.initialize_std_dev), 
            bias_initializer=tf.keras.initializers.TruncatedNormal(0, self.initialize_std_dev))
            for _ in range(self.convs_per_resolution)] for feature_count in self.features]

        
        # For each AdaIN layer, we need an instance normalization layer
        # and a learned affine transformation that will be applied to the
        # style vector.
        self.synthesis_norm = [[tfa.layers.InstanceNormalization() for _ in range(self.convs_per_resolution)] for feature_count in self.features]
        self.affine_transform = [[tf.Variable(tf.random.normal([feature_count, self.map_dimension])) for _ in range(self.convs_per_resolution)] for feature_count in self.features]
        self.affine_bias = [[tf.Variable(tf.random.normal([1, feature_count], seed=self.seed)) for _ in range(self.convs_per_resolution)] for feature_count in self.features]

    @tf.function
    def call(self, input):
        # Generate style vector from latent space input
        # Shape ends up as [batches, map_dimension]
        style = input
        for i in range(self.map_depth):
            style = self.map_activation[i](self.map_dense[i](style))
        
        # Extend the constant synthesis base along the batch dimension.
        base = tf.repeat(self.synthesis_base, input.shape[0], axis=0)
        # base = tf.reshape(style, [input.shape[0], 4, 4, 64])

        # Pass it through the synthesis network.
        output = base
        for i, resolution in enumerate(self.resolutions):
            output = tf.image.resize(output, [resolution, resolution])
            for j in range(self.convs_per_resolution):
                # Convolution
                output = self.synthesis_conv[i][j](output)
                # AdaIn.
                # Apply normalization
                output = self.synthesis_norm[i][j](output)
                # Bias: shape is [batches, # of features]
                bias = tf.repeat(self.affine_bias[i][j], input.shape[0], axis=0)
                # Multiplier: shape is [batches, # of features]
                # Affine transform: shape is [# of features, map_dimension]
                # Style: shape is [batches, map_dimension]
                multiplier = tf.einsum("bm,fm->bf", style, self.affine_transform[i][j])
                # output = multiplier * output + bias, for each batch item, for 
                # each feature
                multiplier = tf.reshape(multiplier, [multiplier.shape[0], 1, 1, multiplier.shape[1]])
                bias = tf.reshape(bias, [bias.shape[0], 1, 1, bias.shape[1]])
                output = multiplier * output + bias

        return output