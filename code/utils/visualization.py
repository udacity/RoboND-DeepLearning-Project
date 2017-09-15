from scipy import misc
import numpy as np

def overlay_predictions(image, im_softmax,image_shape, threshold, channel, seg_color=(0,255,0,172)):
    """creates a overlay using pixels with p(class) > threshold"""

    segmentation = np.expand_dims(im_softmax[:,:,channel] > threshold, 2)
    mask = segmentation * np.reshape(np.array(seg_color), (1,1,-1))
    mask = misc.toimage(mask, mode="RGBA")
    street_im = misc.toimage(image)
    street_im.paste(mask, box=None, mask=mask)
    return street_im
