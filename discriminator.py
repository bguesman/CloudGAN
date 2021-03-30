import tensorflow as tf

"""

This class is where we define the discriminator as an object.

"""

class Discriminator(tf.keras.Model):
    def __init__(self, learning_rate=0.00002):

        # I don't know why this is here, but I'm afraid to touch it.
        ######vvv DO NOT CHANGE vvvv##############
        super(Discriminator, self).__init__()
        ######^^^ DO NOT CHANGE ^^^##################

        # Define the optimizer we want to use to train the model.
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)

        self.bn = tf.keras.layers.BatchNormalization()
        self.c1 = tf.keras.layers.Conv2D(filters=2, kernel_size=3, strides=2, padding='same', activation="relu")
        self.c2 = tf.keras.layers.Conv2D(filters=4, kernel_size=3, strides=2, padding='same', activation="relu")
        self.c3 = tf.keras.layers.Conv2D(filters=8, kernel_size=3, strides=2, padding='same', activation="relu")
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(32, activation="relu")
        self.d2 = tf.keras.layers.Dense(1)

    @tf.function
    def call(self, input):
        # Our input is a batch of N single-channel 400x400 images,
        # so its shape is: (N, 400, 400, 1)
        # We want to output a prediction of real/fake for each one.
        convolved = self.c3(self.c2(self.c1(input)))
        flattened = self.flatten(convolved)
        logits = self.d2(self.d1(flattened))
        return logits

    def loss(self, prediction, ground_truth):
        return tf.reduce_sum(tf.keras.losses.binary_crossentropy(ground_truth, prediction, from_logits=True)) / ground_truth.shape[0]
