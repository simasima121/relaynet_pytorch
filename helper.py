import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imsave
import os

def silentremove(filename):
    '''
    Remove filename
    '''
    try:
        os.remove(filename)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occurred
def remove_files(mydir, ext):
    '''
    Remove all files from a folder
    Inputs are directionry and extension of files to remove
    '''
    filelist = [ f for f in os.listdir(mydir) if f.endswith(ext) ]
    for f in filelist:
        os.remove(os.path.join(mydir, f))

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
    #name = directory + word
    #print(name)
    number = "_" + str(i+1)
    name = directory + "Resized_train/" + word[:to_replace] + number + ext
    if colour == True:
        number = "_colour_" + str(i+1)
        name = directory + "Resized_train/" + word[:to_replace] + number + ext
    imsave(name, image)

def save_result_image(image, directory, word):
    '''
    Save image
    
    Input is image, directory, word, extension, i, colour (Bool - False)
    '''
    name = "_result_raw"
    ext = word[-4:]
    name = directory + word[:-4] + name + '.png'
    imsave(name, image)
    print('Saved',name)

def save_np_array(weighted_image, directory, word, ext, i, ids=False):
    '''
    Save image
    
    Input is image, directory, word, extension, i, colour (Bool - False)
    '''
    to_replace = word.find(ext)
    if ids:
        number = "Train"+ word[:to_replace] + "_id_" + str(i+1)
    else:
        number = "_weight_" + str(i+1)
        
    name = directory + "Train" + word[:to_replace] + number + '.npy'
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
