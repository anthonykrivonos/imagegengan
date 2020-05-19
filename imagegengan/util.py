import os, cv2, random
from os import mkdir, chdir
from os.path import exists
import numpy as np
from enum import Enum
from os import listdir
from os.path import join
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model


class Resizing(Enum):
    CONTAIN = 0
    STRETCH = 1
    COVER = 2


def ImageFlow(img_shape, from_dir=None, from_list=[], from_prepared=[], verbose=True, grayscale=False, resizing=Resizing.COVER, padding_color=0, limit=None, shuffle=False, horizontal_flip=False, batch_size=16, seed=1, rounds=1):
    """
    Get a Keras ImageDataGenerator flow from either images from a directory, images from a list, or images preprepared with the prepare_images function. Only one of such inputs must be provided.
    :param img_shape: The image shape as a three-dimensional tuple (height, width, channels).
    :param from_dir: The directory to train from. The images here can be different sizes and the resizing function will standardize them. Do not supply from_list or from_prepared if you choose to train from a directory. (default None)
    :param from_list: A numpy list containing *unprepared* images to train from. The images here can be different sizes and the resizing function will standardize them. Do not supply from_dir or from_prepared if you choose to train from a list. (default [])
    :param from_prepared: A numpy list or string .npy filename containing *prepared* images using the imagegengan.prepare_images function. The images here *must* be the same size as the image shape of the imagegengan class. Do not supply from_dir or from_list if you choose to train from a prepared list. (default [])
    :param verbose: Boolean indicating whether the flow generation's status should be outputted. (default True)
    :param grayscale: Boolean indicating whether or not the images should be converted into grayscale. (default False)
    :param resizing: The resizing method. Use Resizing.CONTAIN or 0 to pad the images, Resizing.STRETCH or 1 to stretch the images and ignore aspect ratio, and Resizing.COVER or 2 to crop the images while maintaining aspect ratio. (default Resizing.COVER)
    :param padding_color: The color of the padding used if the resizing mode is Resizing.CONTAIN or 0. Defaults to 0, which is black. (default 0)
    :param horizontal_flip: Boolean indicating whether the training images can be horizontally flipped during image data augmentation. This helps the GAN generator more kinds of images. (default False)
    :param seed: Integer seed for the random image augmenter. (default 1)
    :param rounds: Number of rounds to go through during image augmentation. The more rounds, the longer it takes to train, but the more results yielded. (default 1)
    :param limit: Limits the number of training images used, mainly for debugging or retraining purposes. (default None)
    :param shuffle: Boolean indicating whether or not to shuffle the training images. Often helpful during retraining. (default False)
    :return: An ImageDataGenerator flow from the input images.
    """

    if from_prepared is None or len(from_prepared) == 0:
        # Generate prepared images
        images = prepare_images(img_shape, from_dir=from_dir, from_list=from_list, verbose=verbose, grayscale=grayscale, resizing=resizing, padding_color=padding_color, limit=limit, shuffle=shuffle)
    else:
        if isinstance(from_prepared, str):
            # Load from file
            images = load_array(from_prepared)
        else:
            # Copy array
            images = from_prepared.copy()
        if shuffle:
            random.shuffle(images)
        if limit:
            images = images[:limit]
    num_images = len(images)

    vprint(verbose, "Augmenting the image dataset...")

    # Augment the dataset
    default_manipulation_range = 0.2
    image_generator = ImageDataGenerator(
        rotation_range=default_manipulation_range*180,
        width_shift_range=default_manipulation_range,
        height_shift_range=default_manipulation_range,
        shear_range=default_manipulation_range,
        zoom_range=default_manipulation_range,
        horizontal_flip=horizontal_flip,
    )
    image_generator.fit(images, augment=True, seed=seed, rounds=rounds)

    # Trick to get around flow's need for labels: https://github.com/keras-team/keras/issues/2694#issuecomment-218446360
    def flow_generator(X, gen):
        flow = gen.flow(X, np.ones(X.shape[0]), batch_size=batch_size)
        for x, y in flow:
            yield x

    return flow_generator(images, image_generator), num_images


def prepare_images(img_shape, from_dir=None, from_list=[], save_to_npy=None, verbose=True, grayscale=False, resizing=Resizing.COVER, padding_color=0, limit=None, shuffle=False):
    """
    Creates a numpy image list of resized images from either images from a directory or images from a list. Only one of such inputs must be provided.
    :param img_shape: The image shape as a three-dimensional tuple (height, width, channels).
    :param from_dir: The directory to train from. The images here can be different sizes and the resizing function will standardize them. Do not supply from_list or from_prepared if you choose to train from a directory. (default None)
    :param from_list: A numpy list containing *unprepared* images to train from. The images here can be different sizes and the resizing function will standardize them. Do not supply from_dir or from_prepared if you choose to train from a list. (default [])
    :param save_to_npy: A string filename to where the numpy array output can be saved. Saving prepared images to a .npy file and inputting these images into imagegengan.train using from_prepared is the recommended (but not easiest) training method. If None, the output images are not saved. (default None)
    :param verbose: Boolean indicating whether the image preparation status should be outputted. (default True)
    :param grayscale: Boolean indicating whether or not the images should be converted into grayscale. (default False)
    :param resizing: The resizing method. Use Resizing.CONTAIN or 0 to pad the images, Resizing.STRETCH or 1 to stretch the images and ignore aspect ratio, and Resizing.COVER or 2 to crop the images while maintaining aspect ratio. (default Resizing.COVER)
    :param padding_color: The color of the padding used if the resizing mode is Resizing.CONTAIN or 0. Defaults to 0, which is black. (default 0)
    :param limit: Limits the number of training images used, mainly for debugging or retraining purposes. (default None)
    :param shuffle: Boolean indicating whether or not to shuffle the training images. Often helpful during retraining. (default False)
    :return: A numpy array of prepared images that are all the same size.
    """

    # Store the list of images
    images = []

    vprint(verbose, "Reading images of shape %s from %s" % (img_shape, 'list' if from_dir is None else from_dir))

    if from_dir:
        files = listdir(from_dir)
        data_len = len(files)

        # Shuffle
        if shuffle:
            random.shuffle(files)

        # Get the images from the given directory
        for i, filename in enumerate(files):
            image = cv2.imread(join(from_dir, filename), 0 if grayscale else 1)

            if image is not None:
                # Resize the image
                if resizing == Resizing.CONTAIN:
                    image = __resize_contain(image, img_shape, padding_color)
                elif resizing == Resizing.STRETCH:
                    image = __resize_stretch(image, img_shape)
                else:
                    image = __resize_cover(image, img_shape)

                # Include the image
                images.append(image)

            # Verbosity
            vprint(verbose, "%s %s%s" % (filename, progress_bar(i / data_len), " " * 50), end='\r')

            if limit is not None and len(images) == limit:
                break
    else:
        # Shuffle
        if shuffle:
            random.shuffle(from_list)
        from_list = from_list if limit is None else from_list[:limit]
        data_len = len(images)

        for i, image in enumerate(from_list):
            if resizing == Resizing.CONTAIN:
                image = __resize_contain(image, img_shape, padding_color)
            elif resizing == Resizing.STRETCH:
                image = __resize_stretch(image, img_shape)
            else:
                image = __resize_cover(image, img_shape)

            # Include the image
            images.append(image)

            # Verbosity
            vprint(verbose, "Image %d %s%s" % (i + 1, progress_bar(i / data_len), " " * 50), end='\r')

    images = np.array(images)
    data_len = len(images)

    vprint(verbose, "Done. %s%s" % (progress_bar(1), " " * 50))

    if save_to_npy:
        save_array(images, save_to_npy)
        vprint(verbose, "Saved %d prepared images to %s" % (data_len, save_to_npy))

    vprint(verbose, "Done reading images with shape %s from %s" % (images.shape, 'list' if from_dir is None else from_dir))
    
    return images


def __resize_contain(cv2_img, to_shape, padding_color=0):
    """
    Resize the given CV2 image to a square with the given size length, and then pad it with the given color.
    Adapted from https://stackoverflow.com/questions/44720580/resize-image-canvas-to-maintain-square-aspect-ratio-in-python-opencv.
    :param cv2_img: The CV2 image obtained via cv2.imread(...).
    :param to_shape: The desired size int.
    :param padding_color: A color int, list, tuple, or ndarray.
    :return: The padded image.
    """

    # Create height and width variables for better naming
    to_height = to_width = to_shape

    # Get actual image dimensions
    height, width = cv2_img.shape[:2]
    aspect_ratio = width / height

    # Interpolate differently based on the image's relative size
    if height > to_height or width > to_width:
        # Shrink image via inter area as its too large
        interp = cv2.INTER_AREA
    else:
        # Stretch image via inter cubic as its too small
        interp = cv2.INTER_CUBIC

    is_image_horizontal = aspect_ratio > 1
    is_image_vertical = aspect_ratio < 1

    # Height and width we're resizing the image to
    new_height, new_width = to_height, to_width

    # Padding around the new image's inner edges
    pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    if is_image_horizontal:
        # Image is horizontal, so requires vertical padding
        new_height = np.round(new_width / aspect_ratio).astype(int)
        pad_vert = (to_height - new_height) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)

    elif is_image_vertical:
        # Image is vertical, so required horizontal padding
        new_width = np.round(new_height * aspect_ratio).astype(int)
        pad_horiz = (to_width - new_width) / 2
        pad_left, pad_right = np.floor(pad_horiz).astype(int), np.ceil(pad_horiz).astype(int)

    # If only one color is provided and the image is RGB, then set the padding color to an array of length 3
    if len(cv2_img.shape) is 3 and not isinstance(padding_color, (list, tuple, np.ndarray)):
        padding_color = [padding_color] * 3

    # Resize the image to the newly calculated dimensions and interpolation strategy
    new_img = cv2.resize(cv2_img, (new_width, new_height), interpolation=interp)

    # Add the calculated borders around the image
    new_img = cv2.copyMakeBorder(new_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padding_color)

    return new_img


def __resize_stretch(cv2_img, to_shape):
    """
    Resizes the image to the given size, ignoring aspect ratio.
    :param cv2_img: The image to resize.
    :param to_shape: The desired shape.
    :return: A new, resized cv2 image.
    """
    to_width, to_height = to_shape[0], to_shape[1]

    # Resize the image
    new_img = cv2.resize(cv2_img, (to_width, to_height), interpolation=cv2.INTER_AREA)

    return new_img


def __resize_cover(cv2_img, to_shape):
    """
    Crop the given cv2 image to the desired width and height, maintaining the image's original aspect ratio.
    :param cv2_img: The image to crop.
    :param to_shape: The desired shape.
    :return: The new resized and cropped image.
    """
    to_width, to_height = to_shape[0], to_shape[1]

    # Create height and width variables for better naming
    height, width = cv2_img.shape[:2]
    aspect_ratio = width / height

    # Resizing
    max_side = max(to_height, to_width)
    if height < width:
        new_height = max_side
        new_width = int(aspect_ratio * new_height)
    else:
        new_width = max_side
        new_height = int(new_width / aspect_ratio)
    resized_img = cv2.resize(cv2_img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    new_img = crop(resized_img, to_shape)

    return new_img


def crop(image, to_shape):
    """
    Given a numpy array image, crop it to the given shape. The image must be at least the same size as the output shape.
    :param image: The image to crop.
    :param to_shape: At least a two-dimensional image shape tuple.
    :return: The cropped version of the input image as a numpy array.
    """
    height, width = image.shape[0], image.shape[1]
    to_height, to_width = to_shape[0], to_shape[1]
    left_padding = int((width - to_width) / 2)
    right_padding = int(np.ceil((width - to_width) / 2))
    top_padding = int((height - to_height) / 2)
    bottom_padding = int(np.ceil((height - to_height) / 2))

    return np.array(image[top_padding:(height - bottom_padding), left_padding:(width - right_padding)])


def crop_all(images, to_shape):
    """
    Given a list of images, crops them all in sequence using the crop(...) function.
    :param images: The list of images to crop.
    :param to_shape: At least a two-dimensional image shape tuple.
    :return: A list of cropped versions of the input images as a numpy array.
    """
    return np.array([ crop(image, to_shape) for image in images ])


def progress_bar(perc, width = 30):
    """
    Gets a progress bar for printing.
    :param perc: The percent completed.
    :param width: The entire width of the bar.
    :return: The progress bar string.
    """
    assert(width > 10)
    width -= 3
    prog = int(perc * width)
    bar = "[" + "=" * prog + (">" if perc < 1 else "=") + "." * (width - prog) + "]"
    return bar


def vprint(verbose, *args, **kwargs):
    """
    Conditional printing function.
    :param verbose: If true, performs the print.
    :param args: args to print(...).
    :param kwargs: kwargs to print(...).
    """
    if verbose:
        print(*args, **kwargs)


def relative_imwrite(relative_filepath, img):
    """
    Call cv2.imwrite(..., img) on a relative file path.
    :param relative_filepath: The path (i.e. 'processed/train/img-2.jpeg').
    :param img: The cv2 image.
    """

    # Store current working directory so we can navigate back
    cwd = os.getcwd()

    # Get index of last slash to use it as a splitting point
    split_idx = relative_filepath.rindex("/")

    # Create a relative path and filename from this
    relative_path = relative_filepath[:split_idx]
    file_name = relative_filepath[(split_idx + 1):]

    # Extract the directory names
    directory_names = relative_path.split("/")
    top_directory_name = cwd

    # Create the directory if it doesn't exist and then move to it
    # Repeat this for every subdirectory
    for dir_name in directory_names:
        top_directory_name = join(top_directory_name, dir_name)
        if not exists(top_directory_name):
            mkdir(dir_name)
        chdir(dir_name)

    # Save the file at the given path
    file_path = "./" + file_name
    cv2.imwrite(file_path, img)
    chdir(cwd)


def save_array(array, filename):
    """
    Saves a numpy array to a file.
    :param array: The numpy array to save.
    :param filename: The filename to save to.
    """
    np.save(filename, array)


def load_array(filename):
    """
    Loads a numpy array from a file.
    :param filename: The filename to load from.
    :return: The numpy array from the given file.
    """
    if '.npy' not in filename:
        filename += '.npy'
    return np.load(filename, allow_pickle=True)


def freeze_layers(model):
    """
    Freezes all layers in a Keras model, recursively. Used as a workaround to a save bug in Keras.
    :param model: The model to freeze, recursively.
    :return: The model with all layers frozen.
    """
    for i in model.layers:
        i.trainable = False
        if isinstance(i, Model):
            freeze_layers(i)
    return model