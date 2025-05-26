import cv2
import numpy as np
from skimage.morphology import thin, skeletonize, dilation, square
from skimage.morphology import (dilation, erosion, opening, closing, 
    diamond, disk, octagon, rectangle, square, star, cube, ball, 
    octahedron)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def load_image(file_path):
    image = cv2.imread(file_path)
    if image is None:
        print(f"Failed to load image from {file_path}")
    return image

def to_grayscale(image):
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def to_rgb(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        return image
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

def add_noise(image, noise_type, **kwargs):
    if noise_type == 'salt_pepper':
        amount = kwargs.get('amount', 0.01)
        salt_vs_pepper = kwargs.get('salt_vs_pepper', 0.5)
        return _add_salt_pepper_noise(image, amount, salt_vs_pepper)
    elif noise_type == 'gaussian':
        mean = kwargs.get('mean', 0)
        var = kwargs.get('var', 0.01)
        return _add_gaussian_noise(image, mean, var)
    elif noise_type == 'poisson':
        return _add_poisson_noise(image)
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")

def _add_salt_pepper_noise(image, amount, salt_vs_pepper):
    noisy = image.copy()
    row, col = noisy.shape[:2]
    num_salt = np.ceil(amount * row * col * salt_vs_pepper)
    num_pepper = np.ceil(amount * row * col * (1.0 - salt_vs_pepper))
    # Salt noise
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in noisy.shape[:2]]
    if len(noisy.shape) == 2:
        noisy[coords[0], coords[1]] = 255
    else:
        noisy[coords[0], coords[1], :] = 255
    # Pepper noise
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in noisy.shape[:2]]
    if len(noisy.shape) == 2:
        noisy[coords[0], coords[1]] = 0
    else:
        noisy[coords[0], coords[1], :] = 0
    return noisy

def _add_gaussian_noise(image, mean, var):
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, image.shape)
    noisy = image.astype(np.float32) + gaussian * 255
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)

def _add_poisson_noise(image):
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(image.astype(np.float32) * vals / 255.0) / vals * 255
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)

def adjust_brightness(image, value):
    return cv2.convertScaleAbs(image, alpha=1, beta=value)

def adjust_contrast(image, factor):
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)

def compute_histogram(image, plot_height=200, plot_width=260):
    dpi = 100.0
    fig = plt.figure(figsize=(plot_width / dpi, plot_height / dpi), dpi=dpi)
    ax = fig.add_subplot(111)

    fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.90)

    if image is None:
        ax.text(0.5, 0.5, "No image data", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    elif len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        img_to_process = image if len(image.shape) == 2 else image[:,:,0]
        hist = cv2.calcHist([img_to_process], [0], None, [256], [0, 256])
        ax.plot(hist, color='black')
        ax.set_xlim([0, 256])
        ax.set_xlabel("Pixel Intensity")
        ax.set_ylabel("Frequency")
        ax.set_title("Grayscale Histogram")
    else:
        # Color image
        colors = ('b', 'g', 'r') # Matplotlib colors for B, G, R
        channel_names = ('Blue Channel', 'Green Channel', 'Red Channel')
        channels = cv2.split(image)
        all_hists_empty = True
        max_freq = 0
        plotted_hists = []

        for i, (ch, color, name) in enumerate(zip(channels, colors, channel_names)):
            hist = cv2.calcHist([ch], [0], None, [256], [0, 256])
            if np.sum(hist) > 0:
                all_hists_empty = False
                max_freq = max(max_freq, hist.max())
                plotted_hists.append((hist, color, name))
            # ax.plot(hist, color=color, label=name) # Plot individually if not empty
        
        if all_hists_empty:
            ax.clear() # Clear previous empty plots if any
            ax.text(0.5, 0.5, "Empty/Blank Image", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            for hist_data, color, name in plotted_hists:
                ax.plot(hist_data, color=color, label=name)
            ax.set_xlim([0, 256])
            ax.set_ylim([0, max_freq * 1.05]) # Set Y limit based on max frequency
            ax.set_xlabel("Pixel Intensity")
            ax.set_ylabel("Frequency")
            ax.set_title("Color Histogram")
            # ax.legend(loc='upper right') # Legend removed as per request

    fig.tight_layout(rect=[0, 0, 1, 1])

    canvas = FigureCanvas(fig)
    canvas.draw()
    
    # Convert to numpy array
    # buf = canvas.buffer_rgba()
    # img_array_rgba = np.asarray(buf) # For older matplotlib? current returns bytes
    img_array_rgba = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    img_array_rgba = img_array_rgba.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    hist_image_bgr = cv2.cvtColor(img_array_rgba, cv2.COLOR_RGB2BGR)
    
    plt.close(fig) # Important to close the figure
    
    return hist_image_bgr

def equalize_histogram(image):
    if len(image.shape) == 2:
        # Grayscale
        return cv2.equalizeHist(image)
    else:
        # Color image: equalize each channel
        channels = cv2.split(image)
        eq_channels = [cv2.equalizeHist(ch) for ch in channels]
        return cv2.merge(eq_channels)

def low_pass_filter(image, ksize=3):
    return cv2.blur(image, (ksize, ksize))

def high_pass_filter(image):
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)
    if len(image.shape) == 3:
        return cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)
    return laplacian

def median_filter(image, ksize=3):
    return cv2.medianBlur(image, ksize)

def averaging_filter(image, ksize=3):
    return cv2.blur(image, (ksize, ksize))

def laplacian_filter(image):
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return cv2.convertScaleAbs(laplacian)

def gaussian_filter(image, ksize=3):
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (ksize, ksize), 0)

def sobel_vertical(image):
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    return cv2.convertScaleAbs(sobel)

def sobel_horizontal(image):
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return cv2.convertScaleAbs(sobel)

def prewitt_vertical(image):
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[1,0,-1],[1,0,-1],[1,0,-1]], dtype=np.float32)
    prewitt = cv2.filter2D(gray, -1, kernel)
    return prewitt

def prewitt_horizontal(image):
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[1,1,1],[0,0,0],[-1,-1,-1]], dtype=np.float32)
    prewitt = cv2.filter2D(gray, -1, kernel)
    return prewitt

def log_filter(image, ksize=3):
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    log = cv2.Laplacian(blur, cv2.CV_64F)
    return cv2.convertScaleAbs(log)

def canny_edge(image):
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, 100, 200)

def zero_cross(image):
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    log = cv2.Laplacian(gray, cv2.CV_64F)
    zero_cross = np.zeros_like(log, dtype=np.uint8)
    for i in range(1, log.shape[0]-1):
        for j in range(1, log.shape[1]-1):
            patch = log[i-1:i+2, j-1:j+2]
            if (patch.max() > 0 and patch.min() < 0):
                zero_cross[i, j] = 255
    return zero_cross

def thicken(image):
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = (gray > 127).astype(np.uint8)
    thick = dilation(binary, square(3)) * 255
    return thick.astype(np.uint8)

def skeleton(image):
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = (gray > 127).astype(np.uint8)
    skel = skeletonize(binary)
    return (skel * 255).astype(np.uint8)

def thinning(image):
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = (gray > 127).astype(np.uint8)
    thin_img = thin(binary)
    return (thin_img * 255).astype(np.uint8)

def hough_lines(image, plot_height=200, plot_width=260):
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 90, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 80)
    
    result_image_with_lines = image.copy()
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(result_image_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)

    # Create Hough space plot (rho vs theta)
    dpi = 100.0
    fig = plt.figure(figsize=(plot_width / dpi, plot_height / dpi), dpi=dpi)
    ax = fig.add_subplot(111, polar=False) # Polar might be an option, but direct scatter is simpler

    if lines is not None:
        rhos = lines[:,0,0]
        thetas = lines[:,0,1] * 180 / np.pi # Convert theta to degrees for plotting
        ax.scatter(thetas, rhos, marker='.', color='blue')
        ax.set_xlabel("Theta (degrees)")
        ax.set_ylabel("Rho (pixels)")
        ax.set_title("Hough Space (Detected Lines)")
        ax.set_xlim([0, 180]) # Theta range for HoughLines
    else:
        ax.text(0.5, 0.5, "No lines detected", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_xlabel("Theta (degrees)")
        ax.set_ylabel("Rho (pixels)")
        ax.set_title("Hough Space (No Lines)")

    fig.tight_layout()
    canvas = FigureCanvas(fig)
    canvas.draw()
    hough_space_plot_image_rgb = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    hough_space_plot_image_rgb = hough_space_plot_image_rgb.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    hough_space_plot_image_bgr = cv2.cvtColor(hough_space_plot_image_rgb, cv2.COLOR_RGB2BGR)
    plt.close(fig)

    return result_image_with_lines, hough_space_plot_image_bgr

def hough_circles(image, plot_height=200, plot_width=260):
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_gray = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Using a fixed dp for now, might need tuning based on image resolution
    # minDist, param1, param2 are common parameters to tune
    circles = cv2.HoughCircles(blurred_gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=gray.shape[0]//8, #minDist related to image height
                               param1=170, param2=60, minRadius=15, maxRadius=gray.shape[0]//2) #min/maxRadius related to image height

    result_image_with_circles = image.copy()
    if circles is not None:
        circles_detected = np.uint16(np.around(circles))
        for i in circles_detected[0, :]:
            cv2.circle(result_image_with_circles, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(result_image_with_circles, (i[0], i[1]), 2, (0, 0, 255), 3)

    dpi = 100.0
    fig = plt.figure(figsize=(plot_width / dpi, plot_height / dpi), dpi=dpi)
    ax = fig.add_subplot(111)

    if circles is not None:
        centers_x = circles[0, :, 0]
        centers_y = circles[0, :, 1]
        radii = circles[0, :, 2]
        
        normalized_radii = (radii / np.max(radii) if np.max(radii) > 0 else radii) * 100 
        ax.scatter(centers_x, centers_y, s=normalized_radii, alpha=0.6, edgecolors='w', linewidth=0.5)
        ax.set_xlabel("X-coordinate of Center")
        ax.set_ylabel("Y-coordinate of Center")
        ax.set_title(f"Detected Circle Centers ({len(centers_x)} found)")
        ax.set_xlim([0, gray.shape[1]]) # Image width
        ax.set_ylim([gray.shape[0], 0]) # Image height (inverted y-axis for image convention)
        ax.set_aspect('equal', adjustable='box') # Ensure aspect ratio is correct for image coordinates
    else:
        ax.text(0.5, 0.5, "No circles detected", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_xlabel("X-coordinate of Center")
        ax.set_ylabel("Y-coordinate of Center")
        ax.set_title("Detected Circle Centers (None)")
        ax.set_xlim([0, gray.shape[1] if gray is not None else plot_width])
        ax.set_ylim([gray.shape[0] if gray is not None else plot_height, 0])
        ax.set_aspect('equal', adjustable='box')

    fig.tight_layout()
    canvas = FigureCanvas(fig)
    canvas.draw()
    plot_image_rgb = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    plot_image_rgb = plot_image_rgb.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plot_image_bgr = cv2.cvtColor(plot_image_rgb, cv2.COLOR_RGB2BGR)
    plt.close(fig)

    return result_image_with_circles, plot_image_bgr

def get_structuring_element(shape, size, **kwargs):
    if shape == 'diamond':
        return diamond(size)
    elif shape == 'disk':
        return disk(size)
    elif shape == 'octagon':
        return octagon(size, size)
    elif shape == 'rectangle':
        return rectangle(size, kwargs.get('width', size))
    elif shape == 'square':
        return square(size)
    elif shape == 'line':
        # 'selem.line' is not available, fallback to square
        return square(size)
    elif shape == 'pair':
        # 'pairwise' is not available in skimage, fallback to square
        return square(size)
    elif shape == 'periodic':
        # 'periodic_line' is not available in skimage, fallback to square
        return square(size)
    elif shape == 'arbitrary':
        arr = kwargs.get('array', np.ones((size, size), dtype=np.uint8))
        return arr
    else:
        raise ValueError(f"Unsupported structuring element shape: {shape}")

def dilate(image, shape, size, **kwargs):
    selem = get_structuring_element(shape, size, **kwargs)
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = (gray > 127).astype(np.uint8)
    result = dilation(binary, selem) * 255
    return result.astype(np.uint8)

def erode(image, shape, size, **kwargs):
    selem = get_structuring_element(shape, size, **kwargs)
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = (gray > 127).astype(np.uint8)
    result = erosion(binary, selem) * 255
    return result.astype(np.uint8)

def open_morph(image, shape, size, **kwargs):
    selem = get_structuring_element(shape, size, **kwargs)
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = (gray > 127).astype(np.uint8)
    result = opening(binary, selem) * 255
    return result.astype(np.uint8)

def close_morph(image, shape, size, **kwargs):
    selem = get_structuring_element(shape, size, **kwargs)
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = (gray > 127).astype(np.uint8)
    result = closing(binary, selem) * 255
    return result.astype(np.uint8)

def save_image(file_path, image):
    success = cv2.imwrite(file_path, image)
    if not success:
        print(f"Failed to save image to {file_path}")
    return success
