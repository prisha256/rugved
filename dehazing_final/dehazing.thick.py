import cv2
import numpy as np
from matplotlib import pyplot as plt

def get_dark_channel(image, window_size=35):
    min_channel = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    dark_channel = cv2.erode(min_channel, kernel)
    return dark_channel

def get_atmospheric_light(image, dark_channel, top_percent=0.0025):
    flat_dark = dark_channel.ravel()
    flat_image = image.reshape(-1, 3)

    num_pixels = int(max(flat_dark.size * top_percent, 1))
    indices = np.argpartition(flat_dark, -num_pixels)[-num_pixels:]
    A = np.max(flat_image[indices], axis=0)
    return A

def get_transmission(image, A, omega=1, window_size=35):
    norm_image = image / A
    transmission = 1 - omega * get_dark_channel(norm_image, window_size)
    return transmission

def guided_filter(I, p, radius, eps):
    mean_I = cv2.boxFilter(I, cv2.CV_64F, (radius, radius))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (radius, radius))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_64F, (radius, radius))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(I * I, cv2.CV_64F, (radius, radius))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (radius, radius))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (radius, radius))

    q = mean_a * I + mean_b
    return q

def recover_image(image, transmission, A, t0=0.01):
    transmission = np.clip(transmission, t0, 1.0)
    J = (image - A) / transmission[..., np.newaxis] + A
    J = np.clip(J, 0, 1)
    return J

def enhance_contrast(img_rgb):
    img_lab = cv2.cvtColor((img_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_lab)

    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    merged_lab = cv2.merge((cl, a, b))
    enhanced_rgb = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)
    return enhanced_rgb

image_path = "D:\prisha_manipal_sp\sp_rugved\final_taskphase\dehazing_final\001_thick.png"
image_bgr = cv2.imread(image_path)

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
image = image_rgb.astype(np.float64) / 255.0

dark_channel = get_dark_channel(image)
A = get_atmospheric_light(image, dark_channel)
trans = get_transmission(image, A)

gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
refined_trans = guided_filter(gray, trans, radius=40, eps=1e-3)

dehazed = recover_image(image, refined_trans, A)
enhanced = enhance_contrast(dehazed)

plt.figure(figsize=(6, 6))
plt.imshow(enhanced)
plt.title("Dehazed & Enhanced Image")
plt.axis('off')
plt.show()