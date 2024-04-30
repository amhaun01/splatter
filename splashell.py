import numpy as np
import time
import matplotlib.image as image
import matplotlib.pyplot as plt
from numpy.random import rand
import splatter
import imageio.v3 as iio
from skimage.transform import resize
import os

impar = {'NCOL':8}#number of colors/layers
impar['S'] = 1024#image size
#iterations per color
impar['RS'] = 8#filter iterations per layer - more results in more elaborate patterns

impar['rseed'] = int(time.time())
#impar['rseed'] = 1650585833
np.random.seed(impar['rseed'])

immask = .5+.5*1 * iio.imread('wongmask.png')
immask = resize(immask,[impar['S'],impar['S']])
#immask = 1-immask/255
#noisin is the input sample that drives the content
#masking the noise constrains the shape of the output. it's not necessary but can be interesting.
#images in the 'richness' paper were not masked.
noisin = rand(impar['S'],impar['S'])*immask

img, ev = splatter.getSplat(impar,noisin,cmapname='spring')

plt.rcParams["figure.figsize"] = [12, 12]
plt.rcParams["figure.autolayout"] = True

if not os.path.exists('output'):
    os.mkdir('output')

fname = 'img' +'_'+ str(impar['NCOL']) +'_'+ str(impar['RS']) +'_'+ str(impar['rseed']) + 'xx.png'

img[img<0] = 0
img[img>1] = 1
image.imsave('output/' + fname, img)
#image.imsave(impath + 'edge0.png', ev, cmap='gray')

plt.imshow(img)
plt.show()
