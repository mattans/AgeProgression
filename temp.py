import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imsave as ims

def showImagefromBatch(i):
    rec_imgs = sess.run(self.generated_images, feed_dict={self.images: batch})
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(np.reshape(batch[i],[64,64]))
    plt.subplot(1,2,2)
    plt.imshow(rec_imgs[i,:,:,0])

def saveBatchBase():
    reshaped_vis = visualization.reshape(self.batchsize, 128, 128)
    ims("results/base.jpg", merge(reshaped_vis[:64], [8, 8]))

def saveBatchEpoch():
    generated_test = sess.run(self.generated_images, feed_dict={self.images: visualization})
    generated_test = generated_test.reshape(self.batchsize, 128, 128)
    ims("results/" + str(epoch) + ".jpg", merge(generated_test[:64], [8, 8]))




