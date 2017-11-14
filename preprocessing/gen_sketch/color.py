from PIL import Image, ImageEnhance, ImageFilter
from pylab import *
# import numpy as np
from scipy.ndimage import filters
# from skimage import io
import glob, os

in_dir = 'pic_org'
out_dir = 'pic_color'
if not os.path.exists(out_dir): os.mkdir(out_dir)


class MyGaussianBlur(ImageFilter.Filter):
    name = "GaussianBlur"

    def __init__(self, radius=2, bounds=None):
        self.radius = radius
        self.bounds = bounds

    def filter(self, image):
        if self.bounds:
            clips = image.crop(self.bounds).gaussian_blur(self.radius)
            image.paste(clips, self.bounds)
            return image
        else:
            return image.gaussian_blur(self.radius)


def start(mat):
    w, h, c = mat.shape
    ws = int(w * random(1))
    hs = int(h * random(1))
    while (mat[ws, hs, 0] > 254) and (mat[ws, hs, 1] > 254) and (mat[ws, hs, 2] > 254):
        ws = int(w * random(1))
        hs = int(h * random(1))
    return ws, hs


def gen_color_line(mat, dir,max_length,max_dif):
    w, h, c = mat.shape
    ws = int(w * random(1))
    hs = int(h * random(1))
    while (mat[ws, hs, 0] > 254) and (mat[ws, hs, 1] > 254) and (mat[ws, hs, 2] > 254):
        ws = int(w * random(1))
        hs = int(h * random(1))
    if dir == 1:
        wt = ws
        ht = hs
        while (wt < w - 1) and (abs(int(mat[wt, ht, 1]) + int(mat[wt, ht, 2]) + int(mat[wt, ht, 0]) - (
                        int(mat[wt + 1, ht, 1]) + int(mat[wt + 1, ht, 2]) + int(mat[wt + 1, ht, 0]))) < 80):
            wt = wt + 1
    if dir == 2:
        wt = ws
        ht = hs
        while (ht < h - 1) and (abs(int(mat[wt, ht, 1]) + int(mat[wt, ht, 2]) + int(mat[wt, ht, 0]) - (
                        int(mat[wt, ht + 1, 1]) + int(mat[wt, ht + 1, 2]) + int(mat[wt, ht + 1, 0]))) < 3):
            ht = ht + 1
    if dir == 3:
        wt = ws
        ht = hs
        length = 0
        while (length < max_length) and (wt < w-1) and (ht < h-1) and (
            abs(int(mat[wt, ht, 1]) + int(mat[wt, ht, 2]) + int(mat[wt, ht, 0]) - (
                            int(mat[wt + 1, ht + 1, 1]) + int(mat[wt + 1, ht + 1, 2]) + int(
                        mat[wt + 1, ht + 1, 0]))) < max_dif):
            ht += 1
            wt += 1
            length = abs(wt - ws) + abs(ht - hs)
    return ws, hs, wt, ht, length


def main():
    
    count = 0
    #parameter
    wsize = 1024  # double the resolution
    Gamma = 0.97
    Phi = 200
    Epsilon = 0.1
    k = 2
    Sigma = 1.5
    max_length=20
    min_length=10
    max_dif=30
    n_point=50
    dir = 3

    input_paths = glob.glob(in_dir+ '/*.jpg')	
    input_paths+=(glob.glob(in_dir+ '/*.jpeg'))
    input_paths+=(glob.glob(in_dir+ '/*.png'))
    for files1 in input_paths:
        filepath, filename = os.path.split(files1)

        im = Image.open(files1).convert('L')
        im = array(ImageEnhance.Sharpness(im).enhance(3.0))
        im2 = filters.gaussian_filter(im, Sigma)
        im3 = filters.gaussian_filter(im, Sigma * k)
        differencedIm2 = im2 - (Gamma * im3)
        (x, y) = shape(im2)
        for i in range(x):
            for j in range(y):
                if differencedIm2[i, j] < Epsilon:
                    differencedIm2[i, j] = 1
                else:
                    differencedIm2[i, j] = 250 + tanh(Phi * (differencedIm2[i, j]))


        gray_pic = differencedIm2.astype(np.uint8)
        color_pic = Image.open(files1)
        real = np.atleast_2d(color_pic)
        
        
        if real.ndim == 3:
            w, h, c = real.shape
            if c==3:
                image = color_pic.filter(MyGaussianBlur(radius=5))
                mat = np.atleast_2d(image)

                if gray_pic.ndim == 2:
                    gray_pic = np.expand_dims(gray_pic, 2)
                    gray_pic = np.tile(gray_pic, [1, 1, 3])
 


                for i in range(n_point):
                    length = 0
                    while length < min_length:
                        ws, hs, wt, ht, length = gen_color_line(mat, dir,max_length,max_dif)
                    gray_pic[ws:wt, hs:ht, :] = mat[ws:wt, hs:ht, :]

                gray_pic = np.append(real, gray_pic, axis=1)
                final_img = Image.fromarray(gray_pic)

                im = final_img
                w, h = im.size
                hsize = int(h * wsize / float(w))
                if hsize * 2 > wsize:  # crop to three
                    im = im.resize((wsize, hsize))
                    bounds1 = (0, 0, wsize, int(wsize / 2))
                    cropImg1 = im.crop(bounds1)
                    # cropImg1.show()
                    cropImg1.save(os.path.join(out_dir, 'u' + filename))
                    bounds2 = (0, hsize - int(wsize / 2), wsize, hsize)
                    cropImg2 = im.crop(bounds2)
                # cropImg.show()
                    cropImg2.save(os.path.join(out_dir, 'd' + filename))
                else:
                    im = im.resize((wsize, (wsize // 2)))
                    im.save(os.path.join(out_dir, 't' + filename))
                count += 1
                print('done!' + str(count))


                # io.imsave(os.path.join(out_dir, filename), differencedIm2.astype(np.uint8))


if __name__ == '__main__':
    main()
