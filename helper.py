import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imsave

def silentremove(filename):
    '''
    Remove filename
    '''
    try:
        os.remove(filename)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occurred

def rgb_to_grey(rgb):
    '''
    convert rgb color to greyscale
    '''
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def m_of_e(x, base=8):
    '''
    returns multiples of 8
    '''
    return int(base * round(float(x)/base))

def show_main_image(img_data):
    '''
    Show image file after squeezing any unwanted dimensions
    '''
    img_data = np.squeeze(img_data) # we do squeeze image test to remove any additional unwanted dimensions.
    plt.imshow(img_data)
    plt.show()

def crop_image(image,left_bound,right_bound, dimensions = True):
    '''
    Crop image 
    
    Input is: image (array), left bound (int), right bound (int) and dimensions (bool - true)
    '''
    if dimensions:
        image = image[:,left_bound:right_bound]
    else:
        image = image[:, left_bound:right_bound, :]
    return image

def segment_image(image, left_bound, right_bound, split):
    '''
    Segment image 
    
    Segment image by input of image, left bound and right bound, and split int
    '''
    list_of_images = []
    bounds = right_bound - left_bound
    quot, rem = divmod(bounds, split)
    for i in range(quot):
        if len(image.shape) == 2:
            cropped_image = crop_image(image, i*split, (i+1) * split)
            list_of_images.append(cropped_image)
        elif len(image.shape) == 3:
            cropped_image = crop_image(image, i*split, (i+1) * split, False)
            list_of_images.append(cropped_image)
    return list_of_images

def save_image(image, directory, word, ext, i, colour = False):
    '''
    Save image
    
    Input is image, directory, word, extension, i, colour (Bool - False)
    '''
    to_replace = word.find(ext)
    number = "_" + str(i+1)
    name = directory + "Resized_train/" + word[:to_replace] + number + ext
    if colour == True:
        number = "_colour_" + str(i+1)
        name = directory + "Resized_train/" + word[:to_replace] + number + ext
    imsave(name, image)

def save_np_array(weighted_image, directory, word, ext, i, ids=False):
    '''
    Save image
    
    Input is image, directory, word, extension, i, colour (Bool - False)
    '''
    to_replace = word.find(ext)
    if ids:
        number = "_id_" + str(i+1)
    else:
        number = "_weight_" + str(i+1)
        
    name = directory + "Resized_train/" + word[:to_replace] + number + '.npy'
    np.save(name,np.array(weighted_image))

def create_array(n,w,h,asarray=False):
    '''
    Creates an w x h array that can be saved as numpy array
    
    Inputs: number to be put into array, width, height, asarray - numpy array (Bool - set to false)
    '''
    Matrix = [[n for x in range(w)] for y in range(h)]
    if asarray:
        Matrix = np.asarray(Matrix)
    return Matrix
    
def divisorGenerator(n):
    '''
    Finds all divisors of a number
    '''
    divisors = []
    for i in range(1, int(math.sqrt(n) + 1)):
        tmp_divisors = []
        if n % i == 0:
            tmp_divisors.append(i)
            if i*i != n:
                tmp_divisors.append(n / i)
            else:
                tmp_divisors.append(i)
            divisors.append(tmp_divisors)
    return divisors

# Source: https://web.archive.org/web/20170223091206/http://www.johnvinyard.com/blog/?p=268
# Source: https://stackoverflow.com/questions/22685274/divide-an-image-into-5x5-blocks-in-python-and-compute-histogram-for-each-block
from numpy.lib.stride_tricks import as_strided as ast
from itertools import product

def norm_shape(shape):
    '''
    Normalize numpy array shapes so they're always expressed as a tuple, 
    even for one-dimensional shapes.

    Parameters
        shape - an int, or a tuple of ints

    Returns
        a shape tuple
    '''
    try:
        i = int(shape)
        return (i,)
    except TypeError:
        # shape was not a number
        pass

    try:
        t = tuple(shape)
        return t
    except TypeError:
        # shape was not iterable
        pass

    raise TypeError('shape must be an int, or a tuple of ints')

# Sliding window function to segment certain parts of image
def sliding_window(a,ws,ss = None,flatten = False):
    '''
    Return a sliding window over a in any number of dimensions
     
    Parameters:
        a  - an n-dimensional numpy array
        ws - an int (a is 1D) or tuple (a is 2D or greater) representing the size 
             of each dimension of the window
        ss - an int (a is 1D) or tuple (a is 2D or greater) representing the 
             amount to slide the window in each dimension. If not specified, it
             defaults to ws.
        flatten - if True, all slices are flattened, otherwise, there is an 
                  extra dimension for each dimension of the input.
     
    Returns
        an array containing each n-dimensional window from a
    '''
     
    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)
     
    # convert ws, ss, and a.shape to numpy arrays so that we can do math in every 
    # dimension at once.
    ws = np.array(ws)
    ss = np.array(ss)
    shape = np.array(a.shape)
     
     
    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shape),len(ws),len(ss)]
    if 1 != len(set(ls)):
        raise ValueError(\
        'a.shape, ws and ss must all have the same length. They were %s' % str(ls))
     
    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shape):
        raise ValueError(\
        'ws cannot be larger than a in any dimension.\
 a.shape was %s and ws was %s' % (str(a.shape),str(ws)))
     
    # how many slices will there be in each dimension?
    newshape = norm_shape(((shape - ws) // ss) + 1)
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    strided = ast(a,shape = newshape,strides = newstrides)
    if not flatten:
        return strided
     
    # Collapse strided so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    # remove any dimensions with size 1
#     dim = filter(lambda i : i != 1,dim)
    dim = list(filter(lambda i : i != 1,dim))
    return strided.reshape(dim)

# How to use sliding window function above
# rows = 512
# columns = 600
# divisor = 54
# col_size, col_overlap = divmod(columns, divisor)
# row_size, row_overlap = divmod(rows, divisor)
# ws = (row_size, col_size)
# ss = (row_size - row_overlap, col_size - col_overlap)
