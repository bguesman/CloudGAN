import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import cv2
import preprocess
import generator
import discriminator

# k_data_path = "/home/brad/Graphics/CloudGAN/CloudGAN/data/CCSN/unpacked"
k_data_path = "/home/brad/Graphics/CloudGAN/CloudGAN/data/CCSN/CCSN_v2/Ci"
# k_data_path = "/home/brad/Graphics/CloudGAN/CloudGAN/data/CCSN/CCSN_v2/Ac"
k_real = 0.0
k_fake = 1.0

def setup_model():
    return generator.Generator(), discriminator.Discriminator()

def train_batch(generator, discriminator, batch_real, epoch, i):

    ###############################################
    ############# Train discriminator #############
    ###############################################
    with tf.GradientTape() as tape:
        # Generate fake data and concatenate to real, and generate corresponding
        # labels.
        latent_state = tf.random.normal([batch_real.shape[0], generator.latent_dimension])
        batch_fake = generator(latent_state)
        batch = tf.concat([batch_real, batch_fake], axis=0)
        labels = tf.concat([tf.fill((batch_real.shape[0], 1), k_real), 
            tf.fill((batch_real.shape[0], 1), k_fake)] , axis=0)

        # Make a prediction and compute the loss.
        prediction = discriminator(batch)
        loss = discriminator.loss(prediction, labels)
        if (epoch % 2 == 0 and i % 10 == 0):
            print("DISCRIMINATOR LOSS, epoch " + str(epoch) + " batch " + str(i) + ": " + str(loss))
        
    # Apply the gradients.
    gradients = tape.gradient(loss, discriminator.trainable_variables)
    discriminator.optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))

    ###############################################
    ############### Train generator ###############
    ###############################################
    # train the generator for more iterations than the discriminator
    k_generator_iterations = 1
    for k in range(k_generator_iterations):
        with tf.GradientTape() as tape:
            # Generate only fake data.
            latent_state = tf.random.normal([batch_real.shape[0] * 2, generator.latent_dimension])
            batch = generator(latent_state)
            labels = tf.fill((batch.shape[0], 1), k_fake)

            # Make a prediction and compute the loss.
            prediction = discriminator(batch)
            loss = -discriminator.loss(prediction, labels)
        if (k == 0 and epoch % 2 == 0 and i % 10 == 0):
            print("GENERATOR LOSS, epoch " + str(epoch) + " batch " + str(i) + ": " + str(loss))
        
        # Apply the gradients.
        gradients = tape.gradient(loss, generator.trainable_variables)
        generator.optimizer.apply_gradients(zip(gradients, generator.trainable_variables))


def train_epoch(generator, discriminator, real_images, epoch, batch_size=16):
    # Shuffle
    np.random.shuffle(real_images)
    tensor_images = tf.convert_to_tensor(real_images, dtype=tf.float32)
    # tensor_images = tfa.image.rotate(tensor_images, 360.0 * tf.random.uniform([tensor_images.shape[0]]), fill_mode="WRAP")

    # Split into batches.
    num_batches = int(real_images.shape[0] / batch_size)
    for i in range(num_batches):
        train_batch(generator, discriminator,
            tensor_images[i * batch_size:(i + 1) * batch_size, :, :, :], epoch, i)  

def train(generator, discriminator, real_images, test_latent_state, epochs=1):
    for i in range(epochs):
        if i % 10 == 0:
            test(generator, test_latent_state, "test-imgs/test_" + str(i))
        train_epoch(generator, discriminator, real_images, i)
        # TODO: checkpoint

def view(generator, state):
    # Generate a random image
    image = generator(state[0,:,:,:])
    cv2.imwrite("test.png", 255 * np.clip(tf.squeeze(image).numpy(), 0, 1))
    cv2.imshow('Generated', tf.squeeze(image).numpy())
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test(generator, state, path):
    # Generate a random image
    image = generator(state)
    cv2.imwrite(path + "_0.png", 255 * np.clip(tf.squeeze(image[0,:,:,:]).numpy(), 0, 1))
    cv2.imwrite(path + "_1.png", 255 * np.clip(tf.squeeze(image[1,:,:,:]).numpy(), 0, 1))
    cv2.imwrite(path + "_2.png", 255 * np.clip(tf.squeeze(image[2,:,:,:]).numpy(), 0, 1))

def run():
    """
    Runs entirety of model: trains, checkpoints, tests.
    """
    # Get the data.
    images = preprocess.preprocess(k_data_path)

    # Create the model.
    generator, discriminator = setup_model()

    # Global canonical latent state for testing
    test_latent_state = tf.random.normal([3, generator.latent_dimension], seed=1)

    # Train the model
    k_epochs = 5000
    train(generator, discriminator, images, test_latent_state, k_epochs)

    # View an example
    view(generator, test_latent_state)

# Run the script
run()