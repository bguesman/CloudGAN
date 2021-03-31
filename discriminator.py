import tensorflow as tf

"""

This class is where we define the discriminator as an object.

"""

class Discriminator(tf.keras.Model):
    def __init__(self, learning_rate=2e-5):

        # I don't know why this is here, but I'm afraid to touch it.
        ######vvv DO NOT CHANGE vvvv##############
        super(Discriminator, self).__init__()
        ######^^^ DO NOT CHANGE ^^^##################

        # Define the optimizer we want to use to train the model.
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)

        self.initialize_std_dev = 0.1

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.c1 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='same', activation="relu", kernel_initializer=tf.keras.initializers.TruncatedNormal(0, self.initialize_std_dev), bias_initializer=tf.keras.initializers.TruncatedNormal(0, self.initialize_std_dev))
        self.c2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation="relu", kernel_initializer=tf.keras.initializers.TruncatedNormal(0, self.initialize_std_dev), bias_initializer=tf.keras.initializers.TruncatedNormal(0, self.initialize_std_dev))
        self.c3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation="relu", kernel_initializer=tf.keras.initializers.TruncatedNormal(0, self.initialize_std_dev), bias_initializer=tf.keras.initializers.TruncatedNormal(0, self.initialize_std_dev))
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(64, activation="relu", kernel_initializer=tf.keras.initializers.TruncatedNormal(0, self.initialize_std_dev), bias_initializer=tf.keras.initializers.TruncatedNormal(0, self.initialize_std_dev))
        self.d2 = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.TruncatedNormal(0, self.initialize_std_dev), bias_initializer=tf.keras.initializers.TruncatedNormal(0, self.initialize_std_dev))

    @tf.function
    def call(self, input):
        # Our input is a batch of N single-channel 400x400 images,
        # so its shape is: (N, 400, 400, 1)
        # We want to output a prediction of real/fake for each one.
        conv1 = self.bn1(self.c1(input))
        conv2 = self.bn2(self.c2(tf.image.resize(conv1, (128, 128))))
        conv3 = self.c3(tf.image.resize(conv2, (64, 64)))
        flattened = tf.concat([self.flatten(conv1), self.flatten(conv2), self.flatten(conv3)], 1)
        logits = self.d2(self.d1(flattened))
        return logits

    def loss(self, prediction, ground_truth):
        return tf.reduce_sum(tf.keras.losses.binary_crossentropy(ground_truth, prediction, from_logits=True)) / ground_truth.shape[0]
