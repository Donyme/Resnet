import numpy as np

""" This script implements the functions for data augmentation and preprocessing.
"""

def parse_record(record, training=True):
    """ Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [3, 32, 32].
    """
    # Reshape from [depth * height * width] to [depth, height, width].
    depth_major = record.reshape((3, 32, 32))

    # Convert from [depth, height, width] to [height, width, depth]
    image = np.transpose(depth_major, [1, 2, 0])

    image = preprocess_image(image, training)

    # Convert from [height, width, depth] to [depth, height, width]
    image = np.transpose(image, [2, 0, 1])

    return image

def preprocess_image(image, training=True):
    """ Preprocess a single image of shape [height, width, depth].

    Args:
        image: An array of shape [32, 32, 3].
        training: A boolean. Determine whether it is in training mode.
    
    Returns:
        image: An array of shape [32, 32, 3].
    """
    if training:
        ### YOUR CODE HERE
        # Resize the image to add four extra pixels on each side.
        aug_image = np.pad(image, ((4, 4),(4, 4), (0, 0)), 'constant')
        ### YOUR CODE HERE

        ### YOUR CODE HERE
        # Randomly crop a [32, 32] section of the image.
        # HINT: randomly generate the upper left point of the image
        x = np.random.randint(0, aug_image.shape[0]-32, 1)
        y = np.random.randint(0, aug_image.shape[1]-32, 1)
        aug_image = np.array(aug_image)[x[0]:x[0]+32, y[0]:y[0]+32, :]
        ### YOUR CODE HERE

        ### YOUR CODE HERE
        # Randomly flip the image horizontally.
        p = np.random.rand()
        if p > 0.5:
          aug_image = np.flip(aug_image, axis=1)

        image = aug_image[:, :, :]
        ### YOUR CODE HERE

    ### YOUR CODE HERE
    # Subtract off the mean and divide by the standard deviation of the pixels.

    for i in range(image.shape[2]):
      mu    = np.mean(image[:,:,i])
      std   = np.std(image[:,:,i])
      image[:, :, i] = (image[:, :, i] - mu)/std
    ### YOUR CODE HERE

    return image