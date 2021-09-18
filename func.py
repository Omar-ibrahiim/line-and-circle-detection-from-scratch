import random
import numpy as np
import cv2
from scipy import signal
from scipy import fftpack
import matplotlib.pylab as plt


def gray_scale(image):
    b, g, r = image[:,:,0], image[:,:,1], image[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray = gray.astype(np.uint8)
    return gray


# noises
def salt_pepper_noise(img):
    row, col = img.shape
    selected_pixel=random.randint(100,5000)
    for i in range(selected_pixel):
        # set these pixel to white
        x=random.randint(0,col-1)
        y=random.randint(0,row-1)
        img[y][x]=255
    for i in range(selected_pixel):
        # set these pixel to black
        x=random.randint(0,col-1)
        y=random.randint(0,row-1)
        img[y][x]=0
    return img

def gussian_noise(img):
    row, col = img.shape
    mean = 0.0
    std = 15.0
    noise = np.random.normal(mean, std, size=(row, col))
    img_noisy = np.add(img,noise)
    img_noisy = img_noisy.astype(np.uint8)
    return img_noisy


def uniform_noise(img):
    row, col = img.shape
    noise=np.random.uniform(-20, 20, size=(row, col))
    img_noisy = np.add(img, noise)
    img_noisy = img_noisy.astype(np.uint8)
    return img_noisy

#convolution
def apply_mask(img, mask):
    img_masked = signal.convolve2d(img, mask)
    img_masked = img_masked.astype(np.uint8)

    return img_masked

# 3x3
def convolution(img, mask):
    row, col = img.shape
    img_masked= np.zeros([row, col])
    for i in range(1, row - 1):
        for j in range(1, col - 1):
            temp = img[i - 1, j - 1] * mask[0, 0] + img[i - 1, j] * mask[0, 1] + img[i - 1, j + 1] * mask[0, 2]+ img[
                i, j - 1] * mask[1, 0] + img[i, j] * mask[1, 1] + img[i, j + 1] * mask[1, 2] + img[i + 1, j - 1] * mask[
                       2, 0] + img[i + 1, j] * mask[2, 1] + img[i + 1, j + 1] * mask[2, 2]

            img_masked[i, j] = temp

    img_masked = img_masked.astype(np.uint8)
    return img_masked


# filters
# all filters are of size 3x3
def ave_filter(img):
    # row, col = img.shape
    mask = np.ones([3, 3], dtype = int)
    mask = mask/9
    return apply_mask(img, mask)


def gaussian_filter(img, shape):
    sigma = 2.6

    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh

    return apply_mask(img, h)


def median_filter(img):
    row, col = img.shape
    img_masked = np.zeros([row, col])
    for i in range(1, row - 1):
        for j in range(1, col - 1):
            median_array = [img[i - 1, j - 1], img[i - 1, j], img[i - 1, j + 1],
                            img[i, j - 1], img[i, j], img[i, j + 1],
                            img[i + 1, j - 1], img[i + 1, j], img[i + 1, j + 1]]

            img_masked[i, j] = np.median(median_array)

    img_masked = img_masked.astype(np.uint8)
    return img_masked


def laplacian_filter(img):
    mask = np.array([[0, 1, 0],
                     [1, -4, 1],
                     [0, 1, 0]])

    row, col = img.shape
    masked_img = np.zeros([row, col])
    for i in range(1, row - 2):
        for j in range(1, col - 2):
            Ix = np.sum(np.multiply(mask, img[i:i + 3, j:j + 3]))
            masked_img[i + 1, j + 1] = Ix
    masked_img = masked_img.astype(np.uint8)
    return np.uint8(masked_img)


def sobel_edge(img):
    row, col = img.shape
    masked_img = np.zeros([row, col])
    img_dirction = np.zeros([row, col])

    img = gaussian_filter(img, (9, 9))

    kx = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]])
    ky = (kx.transpose())
    for i in range(1, row - 2):
        for j in range(1, col - 2):
            Ix = np.sum(np.multiply(kx, img[i:i + 3, j:j + 3]))
            Iy = np.sum(np.multiply(ky, img[i:i + 3, j:j + 3]))
            masked_img[i + 1, j + 1] = np.hypot(Ix, Iy)
            # calculate the gradient direction
            img_dirction[i + 1, j + 1] = np.arctan2(Iy, Ix)
            img_dirction[i + 1, j + 1] = np.rad2deg(img_dirction[i + 1, j + 1])
            img_dirction[i + 1, j + 1] += 180


    img_dirction = img_dirction.astype(np.uint8)
    masked_img = masked_img.astype(np.uint8)

    return masked_img, img_dirction



def roberts_edge(img):
    # row, col = img.shape
    mask_x = np.array([[0, 0, 0],
                       [0, 1, 0],
                       [0, 0, -1]])

    mask_y = np.array( [[0, 0, 0],
                        [0, 0, 1],
                        [0, -1, 0]])

    blurred_img = gaussian_filter(img, (9,9))
    mask_x_dirc = apply_mask(blurred_img, mask_x)
    mask_y_dirc = apply_mask(blurred_img, mask_y)
    gradient =np.sqrt(np.square(mask_x_dirc) + np.square(mask_y_dirc))
    masked_img = (gradient * 255.0) / gradient.max()

    return np.uint8(masked_img)


def perwitt_edge(img):
    row, col = img.shape
    masked_img = np.zeros([row, col])
    mask_x = np.array([[-1, -1, -1],
                       [0, 0, 0],
                       [1, 1, 1]])
    mask_y = (mask_x.transpose())

    blurred_img = gaussian_filter(img, (9, 9))
    for i in range(1, row - 2):
        for j in range(1, col - 2):
            Ix = np.sum(np.multiply(mask_x, blurred_img[i:i + 3, j:j + 3]))
            Iy = np.sum(np.multiply(mask_y, blurred_img[i:i + 3, j:j + 3]))
            masked_img[i + 1, j + 1] = np.hypot(Ix, Iy)


    masked_img = masked_img.astype(np.uint8)

    return np.uint8(masked_img)


def canny_edge(img):
    row, col = img.shape
    weak = 50
# multi stage filter
# 1) gaussian filter
    blurred_img = gaussian_filter(img, (9,9))

# 2) sobel filter (mag ,dirc)
    gradient_mag, gradient_dirc = sobel_edge(blurred_img)

# 3) Non-max Suppression
    img_non_max_supperession = non_max_suppression(gradient_mag, gradient_dirc)

# 4) Apply thresholding/hysteresis

    image_thresholded = threshold(img_non_max_supperession, 15, 30, weak=weak)
    final_img = hysteresis(image_thresholded, weak)
    final_img = final_img.astype(np.uint8)
    return np.uint8(final_img)


def non_max_suppression(gradient_magnitude, gradient_direction):
    image_row, image_col = gradient_magnitude.shape
    output = np.zeros(gradient_magnitude.shape)
    PI = 180

    for row in range(1, image_row - 1):
        for col in range(1, image_col - 1):
            direction = gradient_direction[row, col]

            if (0 <= direction < PI / 8) or (15 * PI / 8 <= direction <= 2 * PI):
                before_pixel = gradient_magnitude[row, col - 1]
                after_pixel = gradient_magnitude[row, col + 1]

            elif (PI / 8 <= direction < 3 * PI / 8) or (9 * PI / 8 <= direction < 11 * PI / 8):
                before_pixel = gradient_magnitude[row + 1, col - 1]
                after_pixel = gradient_magnitude[row - 1, col + 1]

            elif (3 * PI / 8 <= direction < 5 * PI / 8) or (11 * PI / 8 <= direction < 13 * PI / 8):
                before_pixel = gradient_magnitude[row - 1, col]
                after_pixel = gradient_magnitude[row + 1, col]

            else:
                before_pixel = gradient_magnitude[row - 1, col - 1]
                after_pixel = gradient_magnitude[row + 1, col + 1]

            if gradient_magnitude[row, col] >= before_pixel and gradient_magnitude[row, col] >= after_pixel:
                output[row, col] = gradient_magnitude[row, col]

    return (output)


def threshold(image, low, high, weak):

    output = np.zeros(image.shape)
    strong = 255

    strong_row, strong_col = np.where(image >= high)
    weak_row, weak_col = np.where((image <= high) & (image >= low))

    output[strong_row, strong_col] = strong
    output[weak_row, weak_col] = weak

    return (output)


def hysteresis(image, weak):
    image_row, image_col = image.shape

    top_to_bottom = image.copy()

    for row in range(1, image_row):
        for col in range(1, image_col):
            if top_to_bottom[row, col] == weak:
                if top_to_bottom[row, col + 1] == 255 or top_to_bottom[row, col - 1] == 255 or top_to_bottom[
                    row - 1, col] == 255 or top_to_bottom[
                    row + 1, col] == 255 or top_to_bottom[
                    row - 1, col - 1] == 255 or top_to_bottom[row + 1, col - 1] == 255 or top_to_bottom[
                    row - 1, col + 1] == 255 or top_to_bottom[
                    row + 1, col + 1] == 255:
                    top_to_bottom[row, col] = 255
                else:
                    top_to_bottom[row, col] = 0

    bottom_to_top = image.copy()

    for row in range(image_row - 1, 0, -1):
        for col in range(image_col - 1, 0, -1):
            if bottom_to_top[row, col] == weak:
                if bottom_to_top[row, col + 1] == 255 or bottom_to_top[row, col - 1] == 255 or bottom_to_top[
                    row - 1, col] == 255 or bottom_to_top[
                    row + 1, col] == 255 or bottom_to_top[
                    row - 1, col - 1] == 255 or bottom_to_top[row + 1, col - 1] == 255 or bottom_to_top[
                    row - 1, col + 1] == 255 or bottom_to_top[
                    row + 1, col + 1] == 255:
                    bottom_to_top[row, col] = 255
                else:
                    bottom_to_top[row, col] = 0

    right_to_left = image.copy()

    for row in range(1, image_row):
        for col in range(image_col - 1, 0, -1):
            if right_to_left[row, col] == weak:
                if right_to_left[row, col + 1] == 255 or right_to_left[row, col - 1] == 255 or right_to_left[
                    row - 1, col] == 255 or right_to_left[
                    row + 1, col] == 255 or right_to_left[
                    row - 1, col - 1] == 255 or right_to_left[row + 1, col - 1] == 255 or right_to_left[
                    row - 1, col + 1] == 255 or right_to_left[
                    row + 1, col + 1] == 255:
                    right_to_left[row, col] = 255
                else:
                    right_to_left[row, col] = 0

    left_to_right = image.copy()

    for row in range(image_row - 1, 0, -1):
        for col in range(1, image_col):
            if left_to_right[row, col] == weak:
                if left_to_right[row, col + 1] == 255 or left_to_right[row, col - 1] == 255 or left_to_right[
                    row - 1, col] == 255 or left_to_right[
                    row + 1, col] == 255 or left_to_right[
                    row - 1, col - 1] == 255 or left_to_right[row + 1, col - 1] == 255 or left_to_right[
                    row - 1, col + 1] == 255 or left_to_right[
                    row + 1, col + 1] == 255:
                    left_to_right[row, col] = 255
                else:
                    left_to_right[row, col] = 0

    final_image = top_to_bottom + bottom_to_top + right_to_left + left_to_right
    final_image[final_image > 255] = 255

    return final_image


def element_freq(arr):
    elements_count = {}
    if np.ndim(arr) > 1:
        arr = arr.flatten()
    for element in arr:   # iterating over the elements for frequency
        if element in elements_count: # checking whether it is in the dict or not
            elements_count[element] += 1 # incerementing the count by 1
        else:
            elements_count[element] = 1  # setting the count to 1

    return elements_count

def histogram(img, color):

    element_cumulative_sum = []
    cumulative_temp = 0
    scaled_cumulative_sum = []
    int_scaled_cumulative_sum = []

    # 1)count each pixel
    elements_count = element_freq(img)

    # 2)cumulative summation
    for key in elements_count:
        cumulative_temp = cumulative_temp + elements_count[key]
        element_cumulative_sum.append(cumulative_temp)

    # sanity check
    if cumulative_temp == element_cumulative_sum[-1]:
        print(True)

    # 3)scale
    scale_factor = (255) / (element_cumulative_sum[-1]-element_cumulative_sum[0])

    for i in range(0, len(element_cumulative_sum)):
        scale_element = (element_cumulative_sum[i]-element_cumulative_sum[0]) * scale_factor
        scaled_cumulative_sum.append(scale_element)

    # 4)round the numbers
    for i in range(0, len(scaled_cumulative_sum)):
        int_scaled_cumulative_sum.append(round(scaled_cumulative_sum[i]))
    # sanity check
    if max(int_scaled_cumulative_sum) == 255:
        print(True)

    # 5)draw Histogram
    histogram_elements = element_freq(int_scaled_cumulative_sum)
    plt.bar(histogram_elements.keys(), histogram_elements.values())
    plt.xlabel('Intensity')
    plt.ylabel('Count')

    if color != " ":
        plt.savefig(f'{color}_Histogram.png')
        plt.clf()
    else:
        plt.savefig('Histogram.png')
        plt.clf()


    # return histogram_elements.keys(), histogram_elements.values()


def normalize(img, display):
    Max = np.max(img)
    Min = np.min(img)
    normalized_img = np.array([(x - Min) / (Max - Min) for x in img])
    # sanity check
    if max(normalized_img.flatten()) <= 1 and min(normalized_img.flatten()) >= 0:
        print(True)

    if display == True:
        normalized_img *= 255
        normalized_img = normalized_img.astype(np.uint8)

    return normalized_img


def global_Thresholding(img):
    row, col = img.shape
    threshold_value = 127
    new_img = np.zeros([row, col])
    for i in range(1, row - 1):
        for j in range(1, col - 1):
            if img[i][j] >= threshold_value:
                new_img[i][j] = 255
            else:
                new_img[i][j] = 0

    new_img = new_img.astype(np.uint8)
    return new_img


def local_Thresholding(img):
    row, col = img.shape
    # empty image
    new_img = np.zeros([row, col])
    for i in range(1, row - 5):
        for j in range(1, col - 5):
            # empty mask 11x11
            partial_img_array = [img[i - 5, j - 5], img[i - 5, j - 4], img[i - 5, j - 3], img[i - 5, j - 2], img[i - 5, j - 1],img[i - 5, j], img[i - 5, j + 1], img[i - 5, j + 2], img[i - 5, j + 3], img[i - 5, j + 4], img[i - 5, j + 5],
                                 img[i - 4, j - 5], img[i - 4, j - 4], img[i - 4, j - 3], img[i - 4, j - 2], img[i - 4, j - 1],img[i - 4, j], img[i - 4, j + 1], img[i - 4, j + 2], img[i - 4, j + 3], img[i - 4, j + 4], img[i - 4, j + 5],
                                 img[i - 3, j - 5], img[i - 3, j - 4], img[i - 3, j - 3], img[i - 3, j - 2], img[i - 3, j - 1],img[i - 3, j], img[i - 3, j + 1], img[i - 3, j + 2], img[i - 3, j + 3], img[i - 3, j + 4], img[i - 3, j + 5],
                                 img[i - 2, j - 5], img[i - 2, j - 4], img[i - 2, j - 3], img[i - 2, j - 2], img[i - 2, j - 1],img[i - 2, j], img[i - 2, j + 1], img[i - 2, j + 2], img[i - 2, j + 3], img[i - 2, j + 4], img[i - 2, j + 5],
                                 img[i - 1, j - 5], img[i - 1, j - 4], img[i - 1, j - 3], img[i - 1, j - 2], img[i - 1, j - 1],img[i - 1, j], img[i - 1, j + 1], img[i - 1, j + 2], img[i - 1, j + 3], img[i - 1, j + 4], img[i - 1, j + 5],
                                 img[i, j - 5],     img[i, j - 4],     img[i, j - 3],     img[i, j - 2],     img[i, j - 1],    img[i, j],     img[i, j + 1],     img[i, j + 2],     img[i, j + 3],     img[i, j + 4],     img[i, j + 5],
                                 img[i + 1, j - 5], img[i + 1, j - 4], img[i + 1, j - 3], img[i + 1, j - 2], img[i + 1, j - 1],img[i +1, j],  img[i +1, j + 1],  img[i +1, j + 2],  img[i +1, j + 3],  img[i +1, j + 4],  img[i +1, j + 5],
                                 img[i + 2, j - 5], img[i + 2, j - 4], img[i + 2, j - 3], img[i + 2, j - 2], img[i + 2, j - 1],img[i + 2, j], img[i + 2, j + 1], img[i + 2, j + 2], img[i + 2, j + 3], img[i + 2, j + 4], img[i + 2, j + 5],
                                 img[i + 3, j - 5], img[i + 3, j - 4], img[i + 3, j - 3], img[i + 3, j - 2], img[i + 3, j - 1],img[i + 3, j], img[i + 3, j + 1], img[i + 3, j + 2], img[i + 3, j + 3], img[i + 3, j + 4], img[i + 3, j + 5],
                                 img[i + 4, j - 5], img[i + 4, j - 4], img[i + 4, j - 3], img[i + 4, j - 2], img[i + 4, j - 1],img[i + 4, j], img[i + 4, j + 1], img[i + 4, j + 2], img[i + 4, j + 3], img[i + 4, j + 4], img[i + 4, j + 5],
                                 img[i + 5, j - 5], img[i + 5, j - 4], img[i + 5, j - 3], img[i + 5, j - 2], img[i + 5, j - 1],img[i + 5, j], img[i + 5, j + 1], img[i + 5, j + 2], img[i + 5, j + 3], img[i + 5, j + 4], img[i + 5, j + 5],
                                 ]
            # calculate the mean to be th
            mean_threshold = np.mean(partial_img_array)
            # loop at that part of the image
            for element in (partial_img_array):
                if element >= int(mean_threshold):
                    new_img[i][j] = 255
                else:

                    new_img[i][j] = 0

    new_img = new_img.astype(np.uint8)
    return new_img


def hybrid(image1, image2):
    image_1 = gaussian_filter(image1, (9, 9))
    image_2 = laplacian_filter(image2)
    image_1 = cv2.resize(image_1, (512, 512))
    image_2 = cv2.resize(image_2, (512, 512))

    final = image_1 + image_2

    return np.uint8(final)


def freq_domain_filter(img, filter_type):
    if filter_type == 'lpf':
        mask = np.ones([9, 9], dtype=int)
        mask = mask / 81

    if filter_type == 'hpf':
        mask = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    # width of padding
    width = (img.shape[0] - mask.shape[0], img.shape[1] - mask.shape[1])
    mask = np.pad(mask, (((width[0] + 1) // 2, width[0] // 2), ((width[1] + 1) // 2, width[1] // 2)),
                  'constant')

    mask = fftpack.ifftshift(mask)

    filtered = np.real(fftpack.ifft2(fftpack.fft2(img) * fftpack.fft2(mask)))

    filtered = np.maximum(0, np.minimum(filtered, 255))
    filtered = filtered.astype(np.uint8)
    return filtered


def rgb_histo(image, color):
    if color == 'red':
        img = image[:, :, 2]

    if color == 'green':
        img = image[:, :, 1]

    if color == 'blue':
        img = image[:, :, 0]

    else:
        print('False')


    return histogram(img, color)


## For image equalization##
# distribution frequency
def df(image):
    values = np.zeros(256)
    type(values)
    row, col = image.shape

    for i in range (row):
        for j in range (col):
            # print(values[image[i, j]])
            values[int(image[i, j])] +=1


    return values

# cumulative distribution frequency
def cdf(hist):
    cdf = np.zeros(256)
    cdf[0] = hist[0]

    for i in range(1, 256):
        cdf[i]= cdf[i-1]+hist[i]
    # Now we normalize the histogram
    cdf = [ele*255/cdf[-1] for ele in cdf]
    return cdf

def equalize_image(image):

    hist = df(image)
    my_cdf = cdf(hist)
    image_equalized = np.interp(image, range(0,256), my_cdf)
    image_equalized = image_equalized.astype(np.uint8)
    return image_equalized

