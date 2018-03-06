# %matplotlib inline
import skimage
import skimage.io as io
import skimage.transform 
import sys
import numpy as np
import math
from matplotlib import pyplot
import matplotlib.image as mpimg


IMAGE_LOCATION = '../datasets/blood-cells/dataset2-master/images/TEST/EOSINOPHIL/_0_1616.jpeg'

img = skimage.img_as_float(skimage.io.imread(IMAGE_LOCATION)).astype(np.float32)

# test color reading
# show the original image
pyplot.figure()
pyplot.subplot(1,2,1)
pyplot.imshow(img)
pyplot.axis('on')
pyplot.title('Original image = RGB')

# show the image in BGR - just doing RGB->BGR temporarily for display
imgBGR = img[:, :, (2, 1, 0)]
#pyplot.figure()
pyplot.subplot(1,2,2)
pyplot.imshow(imgBGR)
pyplot.axis('on')
pyplot.title('OpenCV, Caffe2 = BGR')


# Model is expecting 224 x 224, so resize/crop needed.
# Here are the steps we use to preprocess the image.
# (1) Resize the image to 256*256, and crop out the center.
input_height, input_width = 640, 480
print("Model's input shape is: ", str(img.shape))
#print("Original image is %dx%d") % (skimage.)
img299 = skimage.transform.resize(img, (299, 299))
pyplot.figure()
pyplot.imshow(img299)
pyplot.axis('on')
pyplot.title('Resized image to 299x299')
pyplot.show()
print("New image shape:" + str(img299.shape))

# preprocessing to switch from RGB -> BGR
# and swap ordering HWC -> CHW

print ("Image shape before HWC --> CHW conversion: ", img299.shape)
# (1) Since Caffe expects CHW order and the current image is HWC,
#     we will need to change the order.
img299 = img299.swapaxes(1, 2).swapaxes(0, 1)
print ("Image shape after HWC --> CHW conversion: ", img299.shape)

pyplot.figure()
for i in range(3):
    # For some reason, pyplot subplot follows Matlab's indexing
    # convention (starting with 1). Well, we'll just follow it...
    pyplot.subplot(1, 3, i+1)
    pyplot.imshow(img299[i])
    pyplot.axis('off')
    pyplot.title('RGB channel %d' % (i+1))

# (2) Caffe uses a BGR order due to legacy OpenCV issues, so we
#     will change RGB to BGR.
img299 = img299[(2, 1, 0), :, :]
print ("Image shape after BGR conversion: ", img299.shape)
# for discussion later - not helpful at this point
# (3) We will subtract the mean image. Note that skimage loads
#     image in the [0, 1] range so we multiply the pixel values
#     first to get them into [0, 255].
#mean_file = os.path.join(CAFFE_ROOT, 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
#mean = np.load(mean_file).mean(1).mean(1)
#img = img * 255 - mean[:, np.newaxis, np.newaxis]

pyplot.figure()
for i in range(3):
    # For some reason, pyplot subplot follows Matlab's indexing
    # convention (starting with 1). Well, we'll just follow it...
    pyplot.subplot(1, 3, i+1)
    pyplot.imshow(img299[i])
    pyplot.axis('off')
    pyplot.title('BGR channel %d' % (i+1))
# (4) finally, since caffe2 expect the input to have a batch term
#     so we can feed in multiple images, we will simply prepend a
#     batch dimension of size 1. Also, we will make sure image is
#     of type np.float32.

pyplot.show()
img299 = img299[np.newaxis, :, :, :].astype(np.float32)
print ('Final input shape is:', img299.shape)