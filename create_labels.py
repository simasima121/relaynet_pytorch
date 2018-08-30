import numpy as np 

SEG_LABELS_LIST = [
#     {"id": -1, "name": "void", "rgb_values": [0, 0, 0]},
    {"id": 0, "name": "void", "rgb_values": [0, 0, 0]}, # black
    {"id": 1, "name": "Myocardium", "rgb_values": [255,0,0]}, # red
    {"id": 2, "name": "Endocardium", "rgb_values": [0, 0, 255]}, # blue
    {"id": 3, "name": "Fibrosis", "rgb_values": [177,10,255]}, # purple
    {"id": 4, "name": "Fat", "rgb_values": [0, 255, 0]}, # green
    {"id": 5, "name": "Dense Collagen", "rgb_values": [255, 140, 0]}, # orange
    {"id": 6, "name": "Loose Collagen", "rgb_values": [255, 255, 0]}, # yellow
    {"id": 7, "name": "Smooth Muscle", "rgb_values": [255,0,255]}# magenta/pink
]; 

def label_img_to_rgb(label_img):
    label_img = np.squeeze(label_img)
    labels = np.unique(label_img)
    label_infos = [l for l in SEG_LABELS_LIST if l['id'] in labels]

    label_img_rgb = np.array([label_img,
                              label_img,
                              label_img]).transpose(1,2,0)
    for l in label_infos:
        mask = label_img == l['id']
        label_img_rgb[mask] = l['rgb_values']

    return label_img_rgb.astype(np.uint8)


############################# Creating Labels Code ###########################################

def to_rgb(color):
    r,g,b = color
    return (int(r),int(g),int(b))


# Source: https://stackoverflow.com/questions/9694165/convert-rgb-color-to-english-color-name-like-green-with-python
def closest_colour(requested_colour):
    min_colours = {}
    for key, name in wb.css3_hex_to_names.items():
        r_c, g_c, b_c = wb.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = wb.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name

# Source Source: https://stackoverflow.com/questions/45043617/count-the-number-of-objects-of-different-colors-in-an-image-in-python
def find_colors(file_name):
    from skimage import io, morphology, measure
    from sklearn.cluster import KMeans

    img = io.imread(file_name)

    rows, cols, bands = img.shape
    X = img.reshape(rows*cols, bands)


    kmeans = KMeans(n_clusters=6, random_state=0).fit(X)
    labels = kmeans.labels_.reshape(rows, cols)

    for i in np.unique(labels):
        blobs = np.int_(morphology.binary_opening(labels == i))
        color = np.around(kmeans.cluster_centers_[i])
        actual_name, closest_name = get_colour_name(to_rgb(color))
        count = len(np.unique(measure.label(blobs))) - 1
        
        print('Color: {}, RGB: {}  >>  Objects: {}'.format(closest_name,color, count))
        
def pixel_colors(file_name):
    '''
    Creating a dictionary of colours to see what colors are in images.
    '''
    from skimage import io
    if type(file_name) == str:
        img = io.imread(file_name)
    else:
        img = file_name
    
    new_img = np.copy(img)
    rows, cols, bands = new_img.shape
    
    dict_of_colours = {}
    for row in range(150,350):
        if row % 25 == 0:
            print(row)
        for col in range(0,50):
            pixel_color = new_img[row][col]
            actual, close = get_colour_name(pixel_color)
            if actual != None:
                if actual in dict_of_colours:
                    dict_of_colours[actual] += 1
                else:
                    dict_of_colours[actual] = 1
                list_of_colors.add(actual)
            else:
                if close in dict_of_colours:
                    dict_of_colours[close] += 1
                else:
                    dict_of_colours[close] = 1

    return dict_of_colours

#import webcolors as wb
#import numpy as np
#from skimage import io
#from skimage.color import rgb2lab, deltaE_cie76
#from PIL import Image

# Source: https://stackoverflow.com/questions/44428315/similar-color-detection-in-python

# Blurring image 
def blur_image(img, b):
    if b == "average":
        kernel = np.ones((3,6),np.float32)/25
        blurred = cv2.filter2D(img,-1,kernel)
    elif b == "gaussian":
    # Gaussian Blur
        blur=cv2.GaussianBlur(img,(13,13),0)
        blurred=cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
    elif b == "bilateral":
    # Bilaterial Filter
        blur=cv2.bilateralFilter(img,4,25,25)
        blurred=cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
    return blurred

def turn_array(rgb):
    '''
    turn rgb into np.array
    '''
    return np.uint8(np.asarray([[rgb]]))

def turn_rgb_2_lab(color, lab):
    '''
    Convert RGB to CIE 1976 L*a*b*
    '''
    return deltaE_cie76(rgb2lab(color),lab)

def rgb_to_id(rgb, threshold, color, ids, lab):
    '''
    Convert rgb to id
    '''
#     print(get_colour_name(color), color, ids)
    color_3d = turn_array(color)
    dE_color = turn_rgb_2_lab(color_3d, lab)
    rgb[dE_color < threshold] = ids
#     rgb[dE_color < threshold] = color

    return rgb

def ref_colour(rgb):
# def ref_colour():
    '''
    Finding colours within a range of the colour input
    input = pixel
    '''
    rgb = io.imread('https://i.stack.imgur.com/npnrv.png') # to show where the images overlap
    new_rgb = np.copy(rgb)
    
    lab = rgb2lab(new_rgb)
    
    threshold = 15  
    
    # Black = Grey 
    black = [0, 0, 0]
    dimgrey = [105, 105, 105]
    darkslategrey = [47, 79, 79]
    lightgrey = [211, 211, 211]
    gainsboro = [220, 220, 220]
    grey = [128, 128, 128]
    lightslategrey = [119, 136, 153]
    darkgrey = [169, 169, 169]
    
    # NOTE: USED TO BE -1 BUT CHANGED AS VALUE NEEDED FOR CROSS ENTROPY - CAN'T HAVE 0
    rgb_to_id(new_rgb, threshold, black, 0, lab)
    rgb_to_id(new_rgb, threshold, dimgrey, 0, lab)
    rgb_to_id(new_rgb, threshold, darkslategrey, 0, lab)
    rgb_to_id(new_rgb, threshold, lightgrey, 0, lab)
    rgb_to_id(new_rgb, threshold, gainsboro, 0, lab)
    rgb_to_id(new_rgb, threshold, grey, 0, lab)
    rgb_to_id(new_rgb, threshold, lightslategrey, 0, lab)
    rgb_to_id(new_rgb, threshold, darkgrey, 0, lab)
    
    # Red = OrangeRed
    red = [255,0,0]
    orangered = [255, 69, 0]
    firebrick = [181,17,17]
    
    rgb_to_id(new_rgb, threshold, red, 1, lab)
    rgb_to_id(new_rgb, threshold, orangered, 1, lab)
    rgb_to_id(new_rgb, threshold, firebrick, 1, lab)
    
    # Blue
    blue = [0, 0, 255]
    darkslateblue = [72, 61, 139]
    midnightblue = [25, 25, 112]
    mediumblue = [0, 0, 205]
    
    rgb_to_id(new_rgb, threshold, blue, 2, lab)
    rgb_to_id(new_rgb, threshold, darkslateblue, 2, lab)
    rgb_to_id(new_rgb, threshold, midnightblue, 2, lab)
    rgb_to_id(new_rgb, threshold, mediumblue, 2, lab)
    
    # Purple = Dark Violet = Medium Purple = Darkorchid
    purple = [128,0,128]
    darkviolet = [177,10,255]
    mediumpurple = [147, 112, 219]
    darkorchid = [153, 50, 204]
    
    rgb_to_id(new_rgb, threshold, purple, 3, lab)
    rgb_to_id(new_rgb, threshold, darkviolet, 3, lab)
    rgb_to_id(new_rgb, threshold, mediumpurple, 3, lab)
    rgb_to_id(new_rgb, threshold, darkorchid, 3, lab)
    
    # Green is same as Lime and limegreen
    green = [0,128,0]
    lime = [0,255,0]
    limegreen = [50, 205, 50]
    
    rgb_to_id(new_rgb, threshold, green, 4, lab)
    rgb_to_id(new_rgb, threshold, lime, 4, lab)
    rgb_to_id(new_rgb, threshold, limegreen, 4, lab)
    
    # Orange is same as dark orange
    orange = [255, 165, 0]
    darkorange = [255, 140, 0]
    chocolate = [180,101,24]
    
    rgb_to_id(new_rgb, threshold, orange, 5, lab)
    rgb_to_id(new_rgb, threshold, darkorange, 5, lab)
    rgb_to_id(new_rgb, threshold, chocolate, 5, lab)
    
    # Yellow = Gold
    yellow = [255,255,0]
    gold = [255, 215, 0]
    goldenrod = [218, 165, 32]
    goldenrod2 = [183,178,36]
    darkgoldenrod = [184, 134, 11]
    
    rgb_to_id(new_rgb, threshold, yellow, 6, lab)
    rgb_to_id(new_rgb, threshold, gold, 6, lab)
    rgb_to_id(new_rgb, threshold, goldenrod, 6, lab)
    rgb_to_id(new_rgb, threshold, goldenrod2, 6, lab)
    rgb_to_id(new_rgb, threshold, darkgoldenrod, 6, lab)
    
    # Magenta/Fuschia
    magenta = [255,0,255]
    darkmagenta = [139, 0, 139]
    
    magenta_threshold = 20
    
    rgb_to_id(new_rgb, magenta_threshold, magenta, 7, lab)
    rgb_to_id(new_rgb, magenta_threshold, darkmagenta, 7, lab)
    
#     new_grey = new_rgb
    new_grey = cv2.cvtColor( new_rgb, cv2.COLOR_RGB2GRAY ) # convert image to grayscale
    
    return new_grey
# rgb = ref_colour()
# plt.imshow(rgb)

def convert_to_id_image(image):
    '''
    convert rgb images to id
    '''
    returned_grey_image = ref_colour(image)
    rows, cols = returned_grey_image.shape
    g_y = np.copy(returned_grey_image)
    for x in range(0, rows):
        for j in range(0,cols):
#             if returned_grey_image[x][j] > 7: # Setting values in the array to void if they're not in labels
#                 returned_grey_image[x][j] = -1 # this created issue with cross_entropy
            if returned_grey_image[x][j] > 7: # Setting values in the array to void if they're not in labels
                g_y[x][j] = 0
    print(np.unique(g_y))
    return g_y

def convert_to_rgb_image(id_image):
    '''
    Convert id back to rgb 
    '''
    rows, cols = id_image.shape

    black = [0, 0, 0]# id = 0
    red = [255,0,0] # id = 1
    blue = [0, 0, 255] # id = 2
    darkviolet = [177,10,255] # id = 3
    lime = [0,255,0] # id = 4
    darkorange = [255, 140, 0] # id = 5
    yellow = [255, 255, 0] # id = 6
    magenta = [255,0,255] # id = 7
    
    colors = [ black, red, blue, darkviolet, lime, darkorange, yellow, magenta]
    new_image = np.zeros((rows,cols,3))
    for x in range(0, rows):
        for j in range(0,cols):
            pixel_value = int(id_image[x][j])
            if pixel_value > 7: # Setting values in the array to black if they're not in labels
                print(pixel_value)
                new_image[x][j] = black
            else:
                new_image[x][j] = colors[pixel_value] # setting id to colours - values between 1-7
    return new_image.astype(np.uint8)