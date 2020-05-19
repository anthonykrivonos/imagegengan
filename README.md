# 🌄 imagegengan

> A plug-and-play GAN image generator using a Keras/Tensorflow backend. [1]

## ⬇️ Installation

`pip install imagegengan`

## ✨ Features

- Completely plug-and-play DCGAN implementation. [2]
- Takes images *of any dimensions* as inputs and outputs to the dimensions of your choice.
- Supports custom generative and discriminative models, if you need that.
- Built-in image augmentation to mitigate effects of small training datasets.


## 🏃‍♂️ Quick Start

```
from imagegengan import imagegengan

# This is a relative path to a directory containing
# a ton of different-sized images.
images_dir = "dog_images/"

# These are the dimensions of our generated image.
# The third dimension is the # of channels, so use
# 3 for a color (RGB) image and 1 for grayscale.
img_shape = (200, 200, 3)

# Create the image generator.
image_generator = imagegengan(img_shape=img_shape)

# Train the image generator.
image_generator.train(from_dir=images_dir, epochs=150, batch_size=32)

# Generate a new image (outputs a numpy array
# that can be displayed or saved as an image).
generated_image = image_generator.generate()
```

Or, without the comments:

```
from imagegengan import imagegengan

image_generator = imagegengan(img_shape=(200, 200, 3))
image_generator.train(from_dir="dog_images/", epochs=150, batch_size=32)

generated_image = image_generator.generate()
```

## 📖 Documentation

### `imagegenrnn` Class

##### `__init__(...)`

Initializes a new imagegenrnn object.

- **`img_shape`**: The image shape as a three-dimensional tuple (height, width, channels). (default `(224, 224, 3)`)
- **`generator`**: The generator model to use. (default `Generator.Default`)
- **`discriminator`**: The discriminator to use. (default `Discriminator.Default`)
- **`upsample_layers`**: The sequence of layers for upsampling in the generator and downsampling in the discriminator. (default `[1024, 512, 256, 128, 64]`)
- **`kernel_size`**: The kernel size for convolutions in the GAN. 5x5 is the standard so we advise against changing this. (default `(5, 5)`)
- **`noise_depth`**: The length of the input noise vector. 100 is the standard so we advise against changing this. (default `100`)
- **`lr`**: The learning rate of both the generator and discriminator. (default `0.00015`)

#### `train(...)`

> Train the GAN model on either images from a directory, images from a list, or images preprepared with the imagegengan.prepare_images function. Only one of such inputs must be provided.

- **`from_dir`**: The directory to train from. The images here can be different sizes and the resizing function will standardize them. Do not supply from_list or from_prepared if you choose to train from a directory. (default `None`)
- **`from_list`**: A numpy list containing *unprepared* images to train from. The images here can be different sizes and the resizing function will standardize them. Do not supply from_dir or from_prepared if you choose to train from a list. (default `[]`)
- **`from_prepared`**: A numpy list or string .npy filename containing *prepared* images using the imagegengan.prepare_images function. The images here *must* be the same size as the image shape of the imagegengan class. Do not supply from_dir or from_list if you choose to train from a prepared list. (default `[]`)
- **`epochs`**: The number of epochs to train on. (default `100`)
- **`batch_size`**: The size of each training batch. Must be less than or equal to the number of input images. It is advised to keep this number small (~8) to avoid tensor overflows. (default `8`)
- **`lr`**: The learning rate of the GAN (not of the generator or discriminator that were initialized in the constructor). Defaults to the generator's and discriminator's learning rate. (default `None`)
- **`noise_level`**: The coefficient of noise when training the discriminator. (default `0.2`)
- **`save_interval`**: The interval (in epochs) of saving the GAN to a file. If save_to is not None, defaults to 1/10th of the number of epochs. (default `None`)
- **`save_to`**: The directory to save to on the save_interval. If None, the GAN is not saved. (default `None`)
- **`grayscale`**: Boolean indicating whether or not the images should be converted into grayscale. (default `False`)
- **`verbose`**: Boolean indicating whether the training's status should be outputted. (default `True`)
- **`resizing`**: The resizing method. Use Resizing.CONTAIN or 0 to pad the images, Resizing.STRETCH or 1 to stretch the images and ignore aspect ratio, and Resizing.COVER or 2 to crop the images while maintaining aspect ratio. (default `Resizing.COVER`)
- **`padding_color`**: The color of the padding used if the resizing mode is Resizing.CONTAIN or 0. Defaults to 0, which is black. (default `0`)
- **`horizontal_flip`**: Boolean indicating whether the training images can be horizontally flipped during image data augmentation. This helps the GAN generator more kinds of images. (default `False`)
- **`seed`**: Integer seed for the random image augmenter. (default `1`)
- **`rounds`**: Number of rounds to go through during image augmentation. The more rounds, the longer it takes to train, but the more results yielded. (default `1`)
- **`limit`**: Limits the number of training images used, mainly for debugging or retraining purposes. (default `None`)
- **`shuffle`**: Boolean indicating whether or not to shuffle the training images. Often helpful during retraining. (default `False`)

#### `generate(...)`

> Generate a single image or a batch of images in numpy format.

- **`num_outputs`**: The number of images to output. If  1, returns a single image. Otherwise, returns an array of images. (default `1`)
- **`save_to_dir`**: The directory to save the images to, if not None. (default `None`)
- **`file_prefix`**: The prefix of the generated image files. Does not do anything if save_to_dir is None. (default `"generated_img"`)
- **`verbose`**: Boolean indicating whether the status of image generation should be outputted. (default `True`)

**`returns`** A single numpy array output image or a list of such images.

#### `save(...)`

> Save the imagegengan to a .h5 file. Use like `gen.save('myfile')`.

- **`filename`**: The name of the file to save the image as.

#### `load(...)`

> Load the imagegengan from a .h5 file. Use like `gen.load('myfile')`.

- **`filename`**: The name of the file to load the image from.

#### `prepare_images(...)`
_(static method)_

> Creates a numpy image list of resized images from either images from a directory or images from a list. Only one of such inputs must be provided.

- **`img_shape`**: The image shape as a three-dimensional tuple (height, width, channels).
- **`from_dir`**: The directory to train from. The images here can be different sizes and the resizing function will standardize them. Do not supply from_list or from_prepared if you choose to train from a directory. (default `None`)
- **`from_list`**: A numpy list containing *unprepared* images to train from. The images here can be different sizes and the resizing function will standardize them. Do not supply from_dir or from_prepared if you choose to train from a list. (default `[]`)
- **`save_to_npy`**: A string filename to where the numpy array output can be saved. Saving prepared images to a .npy file and inputting these images into imagegengan.train using from_prepared is the recommended (but not easiest) training method. If None, the output images are not saved. (default `None`)
- **`verbose`**: Boolean indicating whether the image preparation status should be outputted. (default `True`)
- **`grayscale`**: Boolean indicating whether or not the images should be converted into grayscale. (default `False`)
- **`resizing`**: The resizing method. Use Resizing.CONTAIN or 0 to pad the images, Resizing.STRETCH or 1 to stretch the images and ignore aspect ratio, and Resizing.COVER or 2 to crop the images while maintaining aspect ratio. (default `Resizing.COVER`)
- **`padding_color`**: The color of the padding used if the resizing mode is Resizing.CONTAIN or 0. Defaults to 0, which is black. (default `0`)
- **`limit`**: Limits the number of training images used, mainly for debugging or retraining purposes. (default None)
- **`shuffle`**: Boolean indicating whether or not to shuffle the training images. Often helpful during retraining. (default `False`)

**`returns`** A numpy array of prepared images that are all the same size.

## 📄 Changelist

#### 1.0.2
- Initial work; used [Tiago Freitas's implementation](https://github.com/tensorfreitas/DCGAN-for-Bird-Generation/blob/master/traindcgan.py), [Mitchell Jolly's implementation](https://github.com/mitchelljy/DCGAN-Keras), and [this issue on ImageDataGenerator flows](https://github.com/keras-team/keras/issues/2694#issuecomment-218446360) for reference.


## 📚 References

[1] Chollet, Francois et al. "Keras." https://keras.io. (2015).

[2] Alec Radford, et al. "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks." (2015).


## Author

Anthony Krivonos ([Portfolio](https://anthonykrivonos.com) | [LinkedIn](https://linkedin.com/in/anthonykrivonos))
