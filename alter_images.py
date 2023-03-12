import cv2
import os

# code to resize images to 256 x 256 and give the images new names (used it on all testing and training images)
desired_size = 256                                                  # set the desired size of the images
count = 0                                                           # set counter for number of images in folder
directory = 'Training/pituitary_tumor'                              # set the directory of the images (changed this for all the image folders)
for filename in os.listdir(directory):
    count += 1                                                      # add 1 to count for every image
    f = os.path.join(directory, filename)                           # create file path
    im = cv2.imread(f)                                              # read in the image
    old_size = im.shape[:2]                                         # get the shape of image
    ratio = float(desired_size)/max(old_size)                       # calculate ratio for new image size
    new_size = tuple([int(x*ratio) for x in old_size])              # calculate size of new image
    im = cv2.resize(im, (new_size[1], new_size[0]))                 # resize image
    delta_w = desired_size - new_size[1]                            # calculate delta between width of image and desired width
    delta_h = desired_size - new_size[0]                            # calculate delta between height of image and desired height
    top, bottom = delta_h//2, delta_h-(delta_h//2)                  # calculate the top and bottom padding to make image a square
    left, right = delta_w//2, delta_w-(delta_w//2)                  # calculate the left and right padding to make image a square
    color = [0, 0, 0]                                               # set color of padding to black
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)     # add padding to make image a square 
    new_filename = f"p({count}).jpg"                                # create new name for image based on the image number
    new_f = os.path.join(directory, new_filename)                   # create image file path with new name
    cv2.imwrite(new_f, new_im)                                      # save new image with new name