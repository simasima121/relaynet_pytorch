def print_hello():
	return "Hello"

def crop_image(image,left_bound,right_bound, dimensions = True):
    if dimensions:
        image = image[:,left_bound:right_bound]
    else:
        image = image[:, left_bound:right_bound, :]
    return image

def segment_image(image, left_bound, right_bound):
    list_of_images = []
    for i in range(9):
        if len(image.shape) == 2:
            cropped_image = crop_image(image, i*64, (i+1) * 64)
            list_of_images.append(cropped_image)
        elif len(image.shape) == 3:
            cropped_image = crop_image(image, i*64, (i+1) * 64, False)
            list_of_images.append(cropped_image)
    return list_of_images

def save_image(image, sub_directory, f, ext, i, colour = False):
    to_replace = f.find(ext)
    number = "_" + str(i+1)
    name = sub_directory + "Resized_train/" + f[:to_replace] + number + ext
    if colour == True:
        number = "_colour_" + str(i+1)
        name = sub_directory + "Resized_train/" + f[:to_replace] + number + ext
