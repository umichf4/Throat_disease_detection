# from pathlib import Path
import cv2
import numpy as np
from matplotlib import pyplot as plt
# import sys


def gradient(x, axis):
    """
        Computes x's gradient along an axis.
    """
    return np.roll(x, -1, axis=axis) - x

def divergence(x, axis):
    """
        Compute x's divergnce along an axis
    """
    return x - np.roll(x, 1, axis=axis)

def erode_dilate(image, struc):
    kernelErosion = np.ones((struc, struc), np.uint8)
    kernelDilation = np.ones((struc, struc), np.uint8)
    # open
    # new_img = cv2.erode(image, kernelErosion, iterations=2)
    # new_img = cv2.dilate(new_img, kernelDilation, iterations=2)
    # close
    new_img = cv2.dilate(image, kernelDilation, iterations=2)
    new_img = cv2.erode(new_img, kernelErosion, iterations=2)

    return new_img

def removeSmallComponents(image, threshold): 
    # find all your connected components (white blobs in your image) 
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8) 
    sizes = stats[1:, -1] 
    nb_components = nb_components - 1 
    img2 = np.zeros(output.shape, dtype=np.uint8) 
    # for every component in the image, you keep it only if it's above threshold 
    for i in range(0, nb_components): 
        if sizes[i] >= threshold: 
            img2[output == i + 1] = 255 
    return img2 

class Image(object):

    def __init__(self):
        """
            Class initializer.
        """
        self.img = None
        self.grayscaled = None
        self.height = 0
        self.width = 0
        self.channel_count = 0

    def open(self, img):
        """
            Opens an image with the name specified.
            Checks if the file exists before opening,
            and throws an error.
        """
#        path = Path(name)
#        if not path.is_file():
#            raise IOError("Could not open image!")

        self.img = img
        self.grayscaled = self.img[:,:,0]
        self.height, self.width, self.channel_count = self.img.shape

    def histogram_equalization(self):
        """
            Corrects over/under exposure in images by using
            histogram equalization.
        """
        img_channels = cv2.split(self.img)
        img_channels[0] = cv2.equalizeHist(img_channels[0])
        self.img = cv2.merge(img_channels)
        return self.img

    def imshow(self):
        '''
            Show the image(BGR)
        '''
        image_RGB = cv2.cvtColor(self.img, cv2.COLOR_YUV2RGB)
        plt.figure("Image")
        plt.imshow(image_RGB)
        plt.show()

    def edge_canny(self, thr_min=50, thr_max=150, thresh=150, struc=5, remove_threshold=200):
        '''
            Apply the Canny alg
            edge_out is the mask
        '''
        blurred = cv2.GaussianBlur(self.img, (3, 3), 0)
        # blurred = cv2.cvtColor(blurred, cv2.COLOR_YUV2BGR)
        # gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        gray = blurred[:,:,0]
        _, gray = cv2.threshold(src=gray, thresh=150, maxval=255, type=cv2.THRESH_TOZERO_INV)
        edge_output = cv2.Canny(gray, thr_min, thr_max)
        edge_output = erode_dilate(edge_output, struc)
        edge_output = removeSmallComponents(edge_output, remove_threshold)
        # cv2.imshow("Canny Edge", edge_output)
        dst = cv2.bitwise_and(self.img, self.img, mask=edge_output)

        return edge_output, dst
    
    def gamma_trans_gray(self, gamma):
        '''
            Apply the gamma transformation in the gray image 
        '''
        # 具体做法是先归一化到1，然后gamma作为指数值求出新的像素值再还原
        gamma_table = [np.power(x/255.0, gamma)*255.0 for x in range(256)]
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
        self.grayscaled = cv2.LUT(self.grayscaled, gamma_table)
        self.img = cv2.merge((self.grayscaled, self.img[:,:,1], self.img[:,:,2]))
        # 实现这个映射用的是OpenCV的查表函数
        return self.img

    def laplace_sharpen(self):
        kernel_sharpen = np.array([
            [-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1]])
        self.img = cv2.filter2D(self.img, -1, kernel_sharpen)
        return self.img

    def balance_channel(self, channel, cutoff=5):
        """
            Applies GIMP's white balancing algorithm.
            Cuts off the low cutoff% values the channel,
            and then stretches the image along the new
            range, creating histogram gaps.
        """

        # low value - cutoff% of the array are lower than this value
        # high value - 100-cutoff% of the array are lower than this value
        #

        low = np.percentile(channel, cutoff)
        high = np.percentile(channel, 100 - cutoff)

        # (high - low) is the new range, basically cutoff% are
        # cut off from each end, because too little pixels have
        # them.
        new_channel = ((channel - low) * 255.0 / (high - low))

        # Convert back to uint8
        channel = np.uint8(np.clip(new_channel, 0, 255))
        return channel

    def balance_white(self, cutoff):
        """
            Wrapper for white balance algorithm
            to balance all channels of an image.
        """
        self.img = cv2.cvtColor(self.img, cv2.COLOR_YUV2BGR)
        b = self.balance_channel(self.img[:,:,0], cutoff)
        g = self.balance_channel(self.img[:,:,1], cutoff)
        r = self.balance_channel(self.img[:,:,2], cutoff)
        self.img = cv2.merge((b,g,r))
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2YUV)
        return self.img

    def denoise_image(self,weight=10, error_tolerance=1e-3):
        """
            Denoises every channel in an image.
        """
        self.img = cv2.cvtColor(self.img, cv2.COLOR_YUV2BGR)
        b = self.denoise_channel(self.img[:, :, 0], weight=weight, error_tolerance=error_tolerance)
        g = self.denoise_channel(self.img[:, :, 1], weight=weight, error_tolerance=error_tolerance)
        r = self.denoise_channel(self.img[:, :, 2], weight=weight, error_tolerance=error_tolerance)
        self.img = cv2.merge((b,g,r))
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2YUV)
        return self.img

    def denoise_channel(self, channel, weight=0.1, error_tolerance=1e-3, iterations=200):
        """
            Denoises a channel using ROF algorithm.
        """
        u = np.zeros(channel.shape)
        px = np.zeros(channel.shape)
        py = np.zeros(channel.shape)

        nm = np.sqrt(np.prod(channel.shape[:2]))
        tau = 0.125

        i = 0
        while i < iterations:
            u_old = u

            # Gradient of U
            ux = gradient(u, 1)
            uy = gradient(u, 0)

            # Update variables
            px_new = px + (tau / weight) * ux
            py_new = py + (tau / weight) * uy

            # Compute p's divergence
            norm = np.maximum(1, np.sqrt(px_new ** 2 + py_new ** 2))
            px = px_new / norm
            py = py_new / norm

            diverg = divergence(px, 1) + divergence(py, 0)
            u = channel + weight * diverg

            # RMSerr
            rms_error = np.linalg.norm(u - u_old) / nm

            if i == 0:
                err_init = rms_error
                err_prev = rms_error
            else:
                # break if error small enough
                if np.abs(err_prev - rms_error) < error_tolerance * err_init:
                    break
                else:
                    err_prev = rms_error

            i += 1

        return u

    



