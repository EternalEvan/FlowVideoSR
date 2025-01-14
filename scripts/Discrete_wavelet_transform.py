import pywt
import numpy as np
from PIL import Image
from scipy.ndimage import zoom


# Load the HQ and LQ images
hq_img_path = "/mnt/nfs/train_REDS4/train_sharp/000/00000000.png"  # Replace with your HQ image path
lq_img_path = "/mnt/nfs/train_REDS4/train_sharp_BICUBIC/000/00000000.png"  # Replace with your LQ image path

hq_img = Image.open(hq_img_path).convert("RGB")
lq_img = Image.open(lq_img_path).convert("RGB")

hq_img_array = np.array(hq_img)
lq_img_array = np.array(lq_img)

# Function to perform DWT on each channel of the RGB image
def dwt2_on_channel(channel):
    coeffs2 = pywt.dwt2(channel, 'haar')
    LL, (LH, HL, HH) = coeffs2
    return LL, LH, HL, HH

# Function to resize LQ subband to the size of HQ subband using scipy's zoom
def resize_lq_to_hq(lq_channel, hq_shape):
    zoom_factors = (hq_shape[0] / lq_channel.shape[0], hq_shape[1] / lq_channel.shape[1])
    resized_channel = zoom(lq_channel, zoom_factors, order=3)  # cubic interpolation (order=3)
    return resized_channel

# Function to perform IDWT to reconstruct each channel
def idwt2_on_channel(LL, LH, HL, HH):
    coeffs2 = (LL, (LH, HL, HH))
    return pywt.idwt2(coeffs2, 'haar')

# Function to merge DWT results into an RGB image for each subband
def merge_channels(r_channel, g_channel, b_channel):
    return np.stack([r_channel, g_channel, b_channel], axis=-1)

# Function to process an image and extract RGB subbands
def process_image(img_array):
    LL_R, LH_R, HL_R, HH_R = dwt2_on_channel(img_array[:, :, 0])  # Red channel
    LL_G, LH_G, HL_G, HH_G = dwt2_on_channel(img_array[:, :, 1])  # Green channel
    LL_B, LH_B, HL_B, HH_B = dwt2_on_channel(img_array[:, :, 2])  # Blue channel

    # Merge subbands into RGB images
    LL_rgb = merge_channels(LL_R, LL_G, LL_B)
    LH_rgb = merge_channels(LH_R, LH_G, LH_B)
    HL_rgb = merge_channels(HL_R, HL_G, HL_B)
    HH_rgb = merge_channels(HH_R, HH_G, HH_B)

    return LL_R, LH_R, HL_R, HH_R, LL_G, LH_G, HL_G, HH_G, LL_B, LH_B, HL_B, HH_B, LL_rgb, LH_rgb, HL_rgb, HH_rgb

# Reconstruct the RGB image using IDWT
def reconstruct_image(LL_R, LH_R, HL_R, HH_R, LL_G, LH_G, HL_G, HH_G, LL_B, LH_B, HL_B, HH_B):
    r_channel_reconstructed = idwt2_on_channel(LL_R, LH_R, HL_R, HH_R)
    g_channel_reconstructed = idwt2_on_channel(LL_G, LH_G, HL_G, HH_G)
    b_channel_reconstructed = idwt2_on_channel(LL_B, LH_B, HL_B, HH_B)
    
    # Merge reconstructed channels into an RGB image
    reconstructed_image = merge_channels(r_channel_reconstructed, g_channel_reconstructed, b_channel_reconstructed)
    return reconstructed_image

# Function to save an image using PIL
def save_image(image_array, file_name):
    image_array_uint8 = np.uint8(image_array)
    img = Image.fromarray(image_array_uint8)
    img.save(f"{file_name}.png", format="PNG")

# Process HQ and LQ images
hq_LL_R, hq_LH_R, hq_HL_R, hq_HH_R, hq_LL_G, hq_LH_G, hq_HL_G, hq_HH_G, hq_LL_B, hq_LH_B, hq_HL_B, hq_HH_B, hq_LL_rgb, hq_LH_rgb, hq_HL_rgb, hq_HH_rgb = process_image(hq_img_array)
lq_LL_R, lq_LH_R, lq_HL_R, lq_HH_R, lq_LL_G, lq_LH_G, lq_HL_G, lq_HH_G, lq_LL_B, lq_LH_B, lq_HL_B, lq_HH_B, lq_LL_rgb, lq_LH_rgb, lq_HL_rgb, lq_HH_rgb = process_image(lq_img_array)

# Reconstruct HQ and LQ images
hq_reconstructed = reconstruct_image(hq_LL_R, hq_LH_R, hq_HL_R, hq_HH_R, hq_LL_G, hq_LH_G, hq_HL_G, hq_HH_G, hq_LL_B, hq_LH_B, hq_HL_B, hq_HH_B)
lq_reconstructed = reconstruct_image(lq_LL_R, lq_LH_R, lq_HL_R, lq_HH_R, lq_LL_G, lq_LH_G, lq_HL_G, lq_HH_G, lq_LL_B, lq_LH_B, lq_HL_B, lq_HH_B)

# Resize LQ subbands to HQ size
resize_lq_LL_R = resize_lq_to_hq(lq_LL_R, hq_LL_R.shape)
resize_lq_LL_G = resize_lq_to_hq(lq_LL_G, hq_LL_G.shape)
resize_lq_LL_B = resize_lq_to_hq(lq_LL_B, hq_LL_B.shape)

# Reconstruct LQ with HQ high-frequency subbands
lq_reconstructed_with_hq = reconstruct_image(resize_lq_LL_R, hq_LH_R, hq_HL_R, hq_HH_R, resize_lq_LL_G, hq_LH_G, hq_HL_G, hq_HH_G, resize_lq_LL_B, hq_LH_B, hq_HL_B, hq_HH_B)

# Resize LQ high-frequency subbands to HQ size
resize_lq_LH_R = resize_lq_to_hq(lq_LH_R, hq_LH_R.shape)
resize_lq_LH_G = resize_lq_to_hq(lq_LH_G, hq_LH_G.shape)
resize_lq_LH_B = resize_lq_to_hq(lq_LH_B, hq_LH_B.shape)
resize_lq_HL_R = resize_lq_to_hq(lq_HL_R, hq_HL_R.shape)
resize_lq_HL_G = resize_lq_to_hq(lq_HL_G, hq_HL_G.shape)
resize_lq_HL_B = resize_lq_to_hq(lq_HL_B, hq_HL_B.shape)
resize_lq_HH_R = resize_lq_to_hq(lq_HH_R, hq_HH_R.shape)
resize_lq_HH_G = resize_lq_to_hq(lq_HH_G, hq_HH_G.shape)
resize_lq_HH_B = resize_lq_to_hq(lq_HH_B, hq_HH_B.shape)

# Reconstruct HQ with resized LQ subbands
hq_reconstructed_with_lq = reconstruct_image(hq_LL_R, resize_lq_LH_R, resize_lq_HL_R, resize_lq_HH_R, hq_LL_G, resize_lq_LH_G, resize_lq_HL_G, resize_lq_HH_G, hq_LL_B, resize_lq_LH_B, resize_lq_HL_B, resize_lq_HH_B)

# Save subband images
save_image(hq_LL_rgb, "HQ_LL_rgb")
save_image(hq_LH_rgb, "HQ_LH_rgb")
save_image(hq_HL_rgb, "HQ_HL_rgb")
save_image(hq_HH_rgb, "HQ_HH_rgb")

save_image(lq_LL_rgb, "LQ_LL_rgb")
save_image(lq_LH_rgb, "LQ_LH_rgb")
save_image(lq_HL_rgb, "LQ_HL_rgb")
save_image(lq_HH_rgb, "LQ_HH_rgb")

# Save original and reconstructed images
save_image(hq_img_array, "HQ_original")
save_image(hq_reconstructed, "HQ_reconstructed")
save_image(lq_img_array, "LQ_original")
save_image(lq_reconstructed, "LQ_reconstructed")

save_image(lq_reconstructed_with_hq, "lq_reconstructed_with_hq")
save_image(hq_reconstructed_with_lq, "hq_reconstructed_with_lq")