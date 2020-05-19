import numpy as np
from os.path import join, exists
from keras.models import load_model

from imagegengan.models import Generator, Discriminator, GAN, compile_model
from imagegengan.util import ImageFlow, progress_bar, vprint, relative_imwrite, crop_all, freeze_layers, prepare_images


class imagegengan:

    def __init__(self, img_shape=(224, 224, 3), generator=Generator.Default, discriminator=Discriminator.Default,
                 upsample_layers=[1024, 512, 256, 128, 64], kernel_size=(5, 5), noise_depth=100, lr=0.00015):
        """
        Initializes a new imagegenrnn object.
        :param img_shape: The image shape as a three-dimensional tuple (height, width, channels). (default (224, 224, 3))
        :param generator: The generator model to use. (default Generator.Default)
        :param discriminator: The discriminator to use. (default Discriminator.Default)
        :param upsample_layers: The sequence of layers for upsampling in the generator and downsampling in the discriminator. (default [1024, 512, 256, 128, 64])
        :param kernel_size: The kernel size for convolutions in the GAN. 5x5 is the standard so we advise against changing this. (default (5, 5))
        :param noise_depth: The length of the input noise vector. 100 is the standard so we advise against changing this. (default 100)
        :param lr: The learning rate of both the generator and discriminator. (default 0.00015)
        """
        self.generator, self.gen_output_shape = generator(img_shape, noise_depth, upsample_layers, kernel_size)
        self.generator = compile_model(self.generator, lr)
        self.discriminator = discriminator(self.gen_output_shape, reversed(upsample_layers), kernel_size)
        self.discriminator = compile_model(self.discriminator, lr)
        self.gan = GAN(noise_depth, self.generator, self.discriminator)
        self.gan = compile_model(self.gan, lr)
        self.lr = lr
        self.img_shape = img_shape
        self.noise_depth = noise_depth

    def train(self, from_dir=None, from_list=[], from_prepared=[],
              epochs=100, batch_size=8, lr=None, noise_level=0.2,
              save_interval=None, save_to=None,
              grayscale=False, verbose=True, resizing=None, padding_color=0,
              horizontal_flip=False, seed=1, rounds=1, limit=None, shuffle=False):
        """
        Train the GAN model on either images from a directory, images from a list, or images preprepared with the imagegengan.prepare_images function. Only one of such inputs must be provided.
        :param from_dir: The directory to train from. The images here can be different sizes and the resizing function will standardize them. Do not supply from_list or from_prepared if you choose to train from a directory. (default None)
        :param from_list: A numpy list containing *unprepared* images to train from. The images here can be different sizes and the resizing function will standardize them. Do not supply from_dir or from_prepared if you choose to train from a list. (default [])
        :param from_prepared: A numpy list or string .npy filename containing *prepared* images using the imagegengan.prepare_images function. The images here *must* be the same size as the image shape of the imagegengan class. Do not supply from_dir or from_list if you choose to train from a prepared list. (default [])
        :param epochs: The number of epochs to train on. (default 100)
        :param batch_size: The size of each training batch. Must be less than or equal to the number of input images. It is advised to keep this number small (~8) to avoid tensor overflows. (default 8)
        :param lr: The learning rate of the GAN (not of the generator or discriminator that were initialized in the constructor). Defaults to the generator's and discriminator's learning rate. (default None)
        :param noise_level: The coefficient of noise when training the discriminator. (default 0.2)
        :param save_interval: The interval (in epochs) of saving the GAN to a file. If save_to is not None, defaults to 1/10th of the number of epochs. (default None)
        :param save_to: The directory to save to on the save_interval. If None, the GAN is not saved. (default None)
        :param grayscale: Boolean indicating whether or not the images should be converted into grayscale. (default False)
        :param verbose: Boolean indicating whether the training's status should be outputted. (default True)
        :param resizing: The resizing method. Use Resizing.CONTAIN or 0 to pad the images, Resizing.STRETCH or 1 to stretch the images and ignore aspect ratio, and Resizing.COVER or 2 to crop the images while maintaining aspect ratio. (default Resizing.COVER)
        :param padding_color: The color of the padding used if the resizing mode is Resizing.CONTAIN or 0. Defaults to 0, which is black. (default 0)
        :param horizontal_flip: Boolean indicating whether the training images can be horizontally flipped during image data augmentation. This helps the GAN generator more kinds of images. (default False)
        :param seed: Integer seed for the random image augmenter. (default 1)
        :param rounds: Number of rounds to go through during image augmentation. The more rounds, the longer it takes to train, but the more results yielded. (default 1)
        :param limit: Limits the number of training images used, mainly for debugging or retraining purposes. (default None)
        :param shuffle: Boolean indicating whether or not to shuffle the training images. Often helpful during retraining. (default False)
        """

        # Assert we have images to train on
        assert (from_dir is not None or len(from_list) > 0 or len(from_prepared) > 0)

        # Load generator
        self.discriminator.trainable = False

        # Create an augmented image flow
        flow, num_images = ImageFlow(self.gen_output_shape, from_dir=from_dir, from_list=from_list,
                                     from_prepared=from_prepared, verbose=verbose, grayscale=grayscale,
                                     resizing=resizing, padding_color=padding_color, horizontal_flip=horizontal_flip,
                                     batch_size=batch_size, seed=seed, rounds=rounds, limit=limit, shuffle=shuffle)

        # Assert we have more images than batch size
        assert (num_images >= batch_size)

        # Compile the GAN with a new LR
        if lr is not None and lr != self.lr:
            self.gan = compile_model(self.gan, lr)

        # Count the number of batches to train for in each epoch
        num_batches = int(num_images / batch_size)

        # Keep track of the losses
        discriminator_losses = []

        # Determine how often to save
        save_interval = save_interval if save_interval is not None and save_interval < epochs else int(epochs / 10)

        vprint(verbose, "Training the image generator...")

        # Train for number of specified epochs
        for e in range(epochs):
            vprint(verbose, "Epoch %d/%d:" % (e + 1, epochs))
            vprint(verbose, "(gan_loss: %1.10f, disc_loss: %1.10f) %s%s" % (
            float('inf'), float('inf'), progress_bar(0), " " * 20), end='\r')

            avg_gan_loss = 0
            avg_d_loss = 0

            # Train per batch
            for b in range(num_batches):
                # Get the next batch of images
                train_images = next(flow)

                # Normalize the images by plotting them on a -1 to 1 scale
                train_images = train_images / (255 / 2) - 1

                # Store the number of training images
                train_images_len = train_images.shape[0]

                # Train GAN
                noise = np.random.normal(0, 1, (train_images_len, self.noise_depth))
                gan_loss = self.gan.train_on_batch(noise, np.ones((train_images_len, 1)))
                avg_gan_loss += gan_loss

                # Generate images
                noise = np.random.normal(0, 1, (train_images_len, self.noise_depth))
                generated_images = self.generator.predict(noise)

                # Add some noise to the labels fed to the discriminator
                train_y = np.ones(train_images_len) - np.random.random_sample(train_images_len) * noise_level
                generated_y = np.random.random_sample(train_images_len) * noise_level

                # Train the discriminator
                self.discriminator.trainable = True
                d_loss = self.discriminator.train_on_batch(train_images, train_y)
                d_loss += self.discriminator.train_on_batch(generated_images, generated_y)
                avg_d_loss += d_loss
                self.discriminator.trainable = False

                # Record the discriminator loss in the history
                discriminator_losses.append(d_loss)

                # Print progress
                vprint(verbose, "(gan_loss: %1.10f, disc_loss: %1.10f) %s%s" % (
                gan_loss, d_loss, progress_bar(b / num_batches), " " * 30), end='\r')

            # Get averages
            avg_gan_loss /= num_batches
            avg_d_loss /= num_batches

            # Print progress
            vprint(verbose,
                   "(gan_loss: %1.10f, disc_loss: %1.10f) %s%s" % (avg_gan_loss, avg_d_loss, progress_bar(1), " " * 20))

            # Save if required
            if save_to is not None and ((e + 1) % save_interval == 0 or e == epochs - 1):
                vprint(verbose, "Saving model to %s..." % save_to)
                self.save(save_to)

    def generate(self, num_outputs=1, save_to_dir=None, file_prefix="generated_img", verbose=True):
        """
        Generate a single image or a batch of images in numpy format.
        :param num_outputs: The number of images to output. If  1, returns a single image. Otherwise, returns an array of images. (default 1)
        :param save_to_dir: The directory to save the images to, if not None. (default None)
        :param file_prefix: The prefix of the generated image files. Does not do anything if save_to_dir is None. (default "generated_img")
        :param verbose: Boolean indicating whether the status of image generation should be outputted. (default True)
        :return: A single numpy array output image or a list of such images.
        """

        # Generate images
        noise = np.random.normal(0, 1, (num_outputs, self.noise_depth))
        generated_images = crop_all(self.generator.predict(noise), self.img_shape)

        # Denormalize the images
        generated_images = np.array((generated_images + 1) * (255 / 2), np.uint8)

        if save_to_dir is not None:
            for i, image in enumerate(generated_images):
                filename = "%s-%d.png" % (file_prefix, i)
                filepath = join(save_to_dir, filename)
                relative_imwrite(filepath, image)
                vprint(verbose, "Wrote generated image to %s." % filepath)

        return generated_images if len(generated_images) != 1 else generated_images[0]

    def save(self, filename):
        """
        Save the imagegengan to a .h5 file. Use like `gen.save('myfile')`.
        :param filename: The name of the file to save the image as.
        """
        frozen_gan = freeze_layers(self.gan)
        if '.h5' not in filename:
            filename += '.h5'
        frozen_gan.save(filename)

    def load(self, filename):
        """
        Load the imagegengan from a .h5 file. Use like `gen.load('myfile')`.
        :param filename: The name of the file to load the image from.
        """
        if '.h5' not in filename:
            filename += '.h5'
        if exists(filename):
            self.gan = load_model(filename)

    @staticmethod
    def prepare_images(*args, **kwargs):
        """
        Mirror of util.prepare_images.
        """
        return prepare_images(*args, **kwargs)