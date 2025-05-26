import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QRadioButton, QButtonGroup,
    QFileDialog, QGroupBox, QGridLayout, QComboBox, QSizePolicy, QScrollArea, QSpinBox, QDoubleSpinBox
)
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor
from PyQt5.QtCore import Qt
import image_utils

class ImageProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Processing Application")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet("background-color: #fafafa;")
        self.original_image = None
        self.noise_image = None
        self.result_image = None
        self.default_kernel_size = 3 # Default kernel size for morphological operations
        self.init_ui()

    def _get_source_image_for_result(self):
        if self.noise_image is not None:
            return self.noise_image
        return self.original_image

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)

        # Left: Functional Sections (now scrollable)
        left_outer_layout = QVBoxLayout() # Renamed from left_layout to avoid confusion
        left_outer_layout.setSpacing(10) # Reduced spacing

        # Create a scroll area for the group boxes
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; }") # Optional: remove scroll area border

        # Create a widget to hold the group boxes and set a layout for it
        scroll_content_widget = QWidget()
        left_group_boxes_layout = QVBoxLayout(scroll_content_widget) # Layout for content inside scroll area
        left_group_boxes_layout.setSpacing(18) # Keep original spacing for group boxes themselves

        left_group_boxes_layout.addWidget(self.create_load_section())
        left_group_boxes_layout.addWidget(self.create_convert_section())
        left_group_boxes_layout.addWidget(self.create_noise_section())
        left_group_boxes_layout.addWidget(self.create_point_section())
        left_group_boxes_layout.addWidget(self.create_local_section())
        left_group_boxes_layout.addWidget(self.create_edge_section())
        left_group_boxes_layout.addWidget(self.create_global_section())
        left_group_boxes_layout.addWidget(self.create_morph_section())
        left_group_boxes_layout.addStretch(1)
        
        scroll_area.setWidget(scroll_content_widget) # Put the content widget into the scroll area

        left_outer_layout.addWidget(scroll_area) # Add scroll area to the main left layout
        left_outer_layout.addLayout(self.create_bottom_controls()) # Bottom controls outside scroll area

        # Right: Image Panels
        right_layout = QVBoxLayout()
        right_layout.setSpacing(18)
        self.img_label_original = self.create_image_panel("Original Image")
        self.img_label_noise = self.create_image_panel("After noise adding")
        self.img_label_result = self.create_image_panel("Result")
        right_layout.addWidget(self.img_label_original)
        right_layout.addWidget(self.img_label_noise)
        right_layout.addWidget(self.img_label_result)

        main_layout.addLayout(left_outer_layout, 2)
        main_layout.addLayout(right_layout, 3)
        self.setCentralWidget(main_widget)

    def create_groupbox(self, title, color):
        box = QGroupBox(title)
        box.setStyleSheet(f"QGroupBox {{ border: 2px solid {color}; border-radius: 8px; margin-top: 8px; }} QGroupBox::title {{ color: red; font-weight: bold; subcontrol-origin: margin; left: 10px; }}")
        return box

    def create_load_section(self):
        box = self.create_groupbox("Load Image", "#FFD600")
        layout = QHBoxLayout()
        btn = QPushButton("Open...")
        btn.setStyleSheet("background: #e0e0e0; font-size: 15px;")
        btn.clicked.connect(self.open_image)
        layout.addWidget(btn)
        box.setLayout(layout)
        return box

    def create_convert_section(self):
        box = self.create_groupbox("Convert", "#FFD600")
        layout = QVBoxLayout()
        self.radio_default = QRadioButton("Default color")
        self.radio_gray = QRadioButton("Gray color")
        self.radio_default.setChecked(True)
        self.convert_color_group = QButtonGroup(self)
        self.convert_color_group.addButton(self.radio_default)
        self.convert_color_group.addButton(self.radio_gray)
        self.radio_default.toggled.connect(lambda checked: self.apply_convert_default(checked))
        self.radio_gray.toggled.connect(lambda checked: self.apply_convert_gray(checked))
        layout.addWidget(self.radio_default)
        layout.addWidget(self.radio_gray)
        box.setLayout(layout)
        return box

    def create_noise_section(self):
        box = self.create_groupbox("Add noise", "#FFD600")
        layout = QVBoxLayout()
        self.radio_sp = QRadioButton("Salt & Pepper noise")
        self.radio_gaussian = QRadioButton("Gaussian noise")
        self.radio_poisson = QRadioButton("Poisson noise")

        self.noise_group = QButtonGroup(self)
        self.noise_group.addButton(self.radio_sp)
        self.noise_group.addButton(self.radio_gaussian)
        self.noise_group.addButton(self.radio_poisson)

        self.radio_sp.toggled.connect(lambda checked: self.apply_noise(checked, 'salt_pepper'))
        self.radio_gaussian.toggled.connect(lambda checked: self.apply_noise(checked, 'gaussian'))
        self.radio_poisson.toggled.connect(lambda checked: self.apply_noise(checked, 'poisson'))

        layout.addWidget(self.radio_sp)
        layout.addWidget(self.radio_gaussian)
        layout.addWidget(self.radio_poisson)

        btn_remove_noise = QPushButton("Remove Applied Noise")
        btn_remove_noise.setStyleSheet("background: #e0e0e0; font-size: 15px;")
        btn_remove_noise.clicked.connect(self.remove_noise_effects)
        layout.addWidget(btn_remove_noise)

        box.setLayout(layout)
        return box

    def create_point_section(self):
        box = self.create_groupbox("Point Transform Ops", "#FF1744")
        layout = QVBoxLayout()

        # Brightness Adjustment
        brightness_layout = QHBoxLayout()
        btn_bright = QPushButton("Brightness adjustment")
        btn_bright.setStyleSheet("background: #e0e0e0; font-size: 15px;")
        self.brightness_spinbox = QSpinBox()
        self.brightness_spinbox.setRange(-255, 255)
        self.brightness_spinbox.setValue(0)
        self.brightness_spinbox.setStyleSheet("font-size: 15px;")
        brightness_layout.addWidget(btn_bright)
        brightness_layout.addWidget(self.brightness_spinbox)
        layout.addLayout(brightness_layout)

        # Contrast Adjustment
        contrast_layout = QHBoxLayout()
        btn_contrast = QPushButton("Contrast adjustment")
        btn_contrast.setStyleSheet("background: #e0e0e0; font-size: 15px;")
        self.contrast_spinbox = QDoubleSpinBox()
        self.contrast_spinbox.setRange(0.1, 3.0)
        self.contrast_spinbox.setSingleStep(0.1)
        self.contrast_spinbox.setValue(1.0)
        self.contrast_spinbox.setStyleSheet("font-size: 15px;")
        contrast_layout.addWidget(btn_contrast)
        contrast_layout.addWidget(self.contrast_spinbox)
        layout.addLayout(contrast_layout)

        # Histogram and Equalization (remain as buttons only)
        btn_hist = QPushButton("Histogram")
        btn_equal = QPushButton("Histogram Equalization")
        for btn in [btn_hist, btn_equal]: # Only these two now for simpler loop
            btn.setStyleSheet("background: #e0e0e0; font-size: 15px;")
            layout.addWidget(btn)
        
        btn_bright.clicked.connect(self.apply_brightness_adjustment)
        btn_contrast.clicked.connect(self.apply_contrast_adjustment)
        # btn_hist is not connected as compute_histogram doesn't return a displayable image for the result panel
        btn_hist.clicked.connect(self.display_result_histogram)
        btn_equal.clicked.connect(self.apply_histogram_equalization)

        box.setLayout(layout)
        return box

    def create_local_section(self):
        box = self.create_groupbox("Local Transform Ops", "#FF1744")
        layout = QVBoxLayout()
        btn_low = QPushButton("Low pass filter")
        btn_high = QPushButton("High pass filter")
        btn_median = QPushButton("Median filtering (gray image)")
        btn_avg = QPushButton("Averaging filtering")
        for btn in [btn_low, btn_high, btn_median, btn_avg]:
            btn.setStyleSheet("background: #e0e0e0; font-size: 15px;")
            layout.addWidget(btn)
        btn_low.clicked.connect(self.apply_low_pass_filter)
        btn_high.clicked.connect(self.apply_high_pass_filter)
        btn_median.clicked.connect(self.apply_median_filter)
        btn_avg.clicked.connect(self.apply_averaging_filter)
        box.setLayout(layout)
        return box

    def create_edge_section(self):
        box = self.create_groupbox("Edge detection filters", "#FFD600")
        layout = QVBoxLayout()
        self.edge_group = QButtonGroup(self)
        edge_filters = [
            "Laplacian filter", "Gaussian filter", "Vert. Sobel", "Horiz. Sobel", "Vert. Prewitt", "Horiz. Prewitt",
            "Lap of Gaussian", "Canny method", "Zero Cross", "Thicken", "Skeleton", "Thinning"
        ]
        for i, name in enumerate(edge_filters):
            radio = QRadioButton(name)
            self.edge_group.addButton(radio, i)
            layout.addWidget(radio)
        self.edge_group.buttonClicked[int].connect(self.apply_edge_filter)
        box.setLayout(layout)
        return box

    def create_global_section(self):
        box = self.create_groupbox("Global Transform Ops", "#FF1744")
        layout = QVBoxLayout()
        btn_line = QPushButton("Line detection using Hough Transform")
        btn_circle = QPushButton("Circles detection using Hough Transform")
        for btn in [btn_line, btn_circle]:
            btn.setStyleSheet("background: #e0e0e0; font-size: 15px;")
            layout.addWidget(btn)
        btn_line.clicked.connect(self.apply_hough_lines)
        btn_circle.clicked.connect(self.apply_hough_circles)
        box.setLayout(layout)
        return box

    def create_morph_section(self):
        box = self.create_groupbox("Morphological Ops", "#FFD600")
        layout = QVBoxLayout()
        btn_dilate = QPushButton("Dilation")
        btn_erode = QPushButton("Erosion")
        btn_close = QPushButton("Close")
        btn_open = QPushButton("Open")
        for btn in [btn_dilate, btn_erode, btn_close, btn_open]:
            btn.setStyleSheet("background: #e0e0e0; font-size: 15px;")
            layout.addWidget(btn)
        btn_dilate.clicked.connect(self.apply_dilation)
        btn_erode.clicked.connect(self.apply_erosion)
        btn_close.clicked.connect(self.apply_closing)
        btn_open.clicked.connect(self.apply_opening)
        self.kernel_combo = QComboBox()
        self.kernel_combo.addItems(["arbitrary", "diamond", "disk", "line", "octagon", "pair", "periodic", "rectangle", "square"])
        self.kernel_combo.setCurrentText("arbitrary")
        self.kernel_combo.setStyleSheet("background: #e0e0e0; font-size: 15px;")
        layout.addWidget(QLabel("Choose type of Kernel:"))
        layout.addWidget(self.kernel_combo)
        box.setLayout(layout)
        return box

    def create_bottom_controls(self):
        layout = QHBoxLayout()
        btn_save = QPushButton("Save Result image")
        btn_exit = QPushButton("Exit")
        
        btn_save.setStyleSheet("background: #e0e0e0; font-size: 15px;")
        btn_exit.setStyleSheet("background: #e0e0e0; font-size: 15px;")
        
        btn_save.clicked.connect(self.save_result_image)
        btn_exit.clicked.connect(self.close)
        
        layout.addWidget(btn_save)
        layout.addWidget(btn_exit)
        return layout

    def create_image_panel(self, title):
        panel = QWidget()
        panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        vbox = QVBoxLayout(panel)
        
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: red; font-weight: bold; font-size: 16px;")
        
        img_label = QLabel()
        img_label.setMinimumSize(150, 150)
        img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        img_label.setStyleSheet("border: 2px solid #FFD600; background: #fff;")
        img_label.setAlignment(Qt.AlignCenter)
        
        vbox.addWidget(title_label)
        vbox.addWidget(img_label)
        
        panel.img_label = img_label
        return panel

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            img = image_utils.load_image(file_path)
            if img is not None:
                self.original_image = img
                self.noise_image = None
                self.result_image = None
                self.display_image(self.img_label_original.img_label, self.original_image)
                self.display_image(self.img_label_noise.img_label, self.noise_image)
                self.display_image(self.img_label_result.img_label, self.result_image)
                if self.radio_default.isChecked():
                    self.apply_convert_default(True)

    def display_image(self, label, img):
        if img is None:
            label.clear()
            return
        if len(img.shape) == 2:
            qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_Grayscale8)
        else:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb_img.data, rgb_img.shape[1], rgb_img.shape[0], rgb_img.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(label.width(), label.height(), Qt.KeepAspectRatio)
        label.setPixmap(pixmap)

    def apply_convert_default(self, checked):
        if not checked or self.original_image is None:
            return
        source_img = self._get_source_image_for_result()
        if source_img is None: return

        if len(source_img.shape) == 2: # Grayscale
            self.result_image = image_utils.to_rgb(source_img)
        else: # Already BGR/RGB
            self.result_image = source_img.copy()
        self.display_image(self.img_label_result.img_label, self.result_image)

    def apply_convert_gray(self, checked):
        if not checked:
            return
        source_img = self._get_source_image_for_result()
        if source_img is None: return
        
        self.result_image = image_utils.to_grayscale(source_img)
        self.display_image(self.img_label_result.img_label, self.result_image)

    def apply_noise(self, checked, noise_type):
        if not checked or self.original_image is None:
            return

        try:
            if noise_type == 'salt_pepper':
                self.noise_image = image_utils.add_noise(self.original_image, noise_type, amount=0.05, salt_vs_pepper=0.5)
            elif noise_type == 'gaussian':
                self.noise_image = image_utils.add_noise(self.original_image, noise_type, mean=0, var=0.01)
            elif noise_type == 'poisson':
                self.noise_image = image_utils.add_noise(self.original_image, noise_type)
            
            self.display_image(self.img_label_noise.img_label, self.noise_image)
            self.result_image = None
            self.display_image(self.img_label_result.img_label, self.result_image)

        except Exception as e:
            print(f"Error applying noise {noise_type}: {e}")
            self.noise_image = None
            self.display_image(self.img_label_noise.img_label, self.noise_image)

    def apply_brightness_adjustment(self):
        source_img = self._get_source_image_for_result()
        if source_img is None: return
        brightness_value = self.brightness_spinbox.value()
        self.result_image = image_utils.adjust_brightness(source_img, brightness_value)
        self.display_image(self.img_label_result.img_label, self.result_image)

    def apply_contrast_adjustment(self):
        source_img = self._get_source_image_for_result()
        if source_img is None: return
        contrast_factor = self.contrast_spinbox.value()
        self.result_image = image_utils.adjust_contrast(source_img, contrast_factor)
        self.display_image(self.img_label_result.img_label, self.result_image)

    def display_result_histogram(self):
        if self.result_image is None:
            # If there's no result image, maybe show histogram of source or a message?
            # For now, let's try to compute from source if result is None
            source_for_hist = self._get_source_image_for_result()
            if source_for_hist is None:
                print("No image available to compute histogram.")
                # Optionally, display a placeholder image in result panel saying "No image"
                # Use current label dimensions for placeholder too
                label_width = self.img_label_result.img_label.width()
                label_height = self.img_label_result.img_label.height()
                label_width = max(label_width, 100) # Min width
                label_height = max(label_height, 100) # Min height
                placeholder_hist_img = image_utils.compute_histogram(None, plot_height=label_height, plot_width=label_width)
                self.display_image(self.img_label_result.img_label, placeholder_hist_img)
                return
        else:
            source_for_hist = self.result_image

        # Get current dimensions of the display label
        label_width = self.img_label_result.img_label.width()
        label_height = self.img_label_result.img_label.height()
        # Ensure minimum dimensions
        label_width = max(label_width, 100) 
        label_height = max(label_height, 100)

        histogram_plot_image = image_utils.compute_histogram(source_for_hist, plot_height=label_height, plot_width=label_width)
        self.display_image(self.img_label_result.img_label, histogram_plot_image)

    def apply_histogram_equalization(self):
        source_img = self._get_source_image_for_result()
        if source_img is None: return
        self.result_image = image_utils.equalize_histogram(source_img)
        self.display_image(self.img_label_result.img_label, self.result_image)

    def apply_low_pass_filter(self):
        source_img = self._get_source_image_for_result()
        if source_img is None: return
        self.result_image = image_utils.low_pass_filter(source_img, ksize=5)
        self.display_image(self.img_label_result.img_label, self.result_image)

    def apply_high_pass_filter(self):
        source_img = self._get_source_image_for_result()
        if source_img is None: return
        self.result_image = image_utils.high_pass_filter(source_img)
        self.display_image(self.img_label_result.img_label, self.result_image)

    def apply_median_filter(self):
        source_img = self._get_source_image_for_result()
        if source_img is None: return
        self.result_image = image_utils.median_filter(source_img, ksize=5)
        self.display_image(self.img_label_result.img_label, self.result_image)

    def apply_averaging_filter(self):
        source_img = self._get_source_image_for_result()
        if source_img is None: return
        self.result_image = image_utils.averaging_filter(source_img, ksize=5)
        self.display_image(self.img_label_result.img_label, self.result_image)

    def apply_edge_filter(self, button_id):
        source_img = self._get_source_image_for_result()
        if source_img is None: return

        edge_functions = [
            image_utils.laplacian_filter, image_utils.gaussian_filter,
            image_utils.sobel_vertical, image_utils.sobel_horizontal,
            image_utils.prewitt_vertical, image_utils.prewitt_horizontal,
            image_utils.log_filter, image_utils.canny_edge,
            image_utils.zero_cross, image_utils.thicken,
            image_utils.skeleton, image_utils.thinning
        ]
        
        if 0 <= button_id < len(edge_functions):
            try:
                func = edge_functions[button_id]
                if func == image_utils.gaussian_filter:
                     self.result_image = func(source_img, ksize=5)
                else:
                    self.result_image = func(source_img)
                self.display_image(self.img_label_result.img_label, self.result_image)
            except Exception as e:
                print(f"Error applying edge filter {self.edge_group.button(button_id).text()}: {e}")

    def apply_hough_lines(self):
        source_img = self._get_source_image_for_result()
        if source_img is None: return

        # Get dimensions for the plot from the noise panel label
        plot_label_width = self.img_label_noise.img_label.width()
        plot_label_height = self.img_label_noise.img_label.height()
        plot_label_width = max(plot_label_width, 100)  # Min width
        plot_label_height = max(plot_label_height, 100) # Min height

        image_with_lines, hough_space_plot = image_utils.hough_lines(source_img, 
                                                                     plot_height=plot_label_height, 
                                                                     plot_width=plot_label_width)
        self.result_image = image_with_lines
        self.display_image(self.img_label_result.img_label, self.result_image)
        
        # Display the hough space plot in the noise panel
        # Note: self.noise_image is not updated, we are just using the label for display.
        self.display_image(self.img_label_noise.img_label, hough_space_plot)

    def apply_hough_circles(self):
        source_img = self._get_source_image_for_result()
        if source_img is None: return

        # Get dimensions for the plot from the noise panel label
        plot_label_width = self.img_label_noise.img_label.width()
        plot_label_height = self.img_label_noise.img_label.height()
        plot_label_width = max(plot_label_width, 100)  # Min width
        plot_label_height = max(plot_label_height, 100) # Min height

        image_with_circles, circle_plot = image_utils.hough_circles(source_img,
                                                                    plot_height=plot_label_height,
                                                                    plot_width=plot_label_width)
        self.result_image = image_with_circles
        self.display_image(self.img_label_result.img_label, self.result_image)
        
        # Display the circle plot in the noise panel
        # Note: self.noise_image is not updated, we are just using the label for display.
        self.display_image(self.img_label_noise.img_label, circle_plot)

    def _apply_morph_op(self, operation_func):
        source_img = self._get_source_image_for_result()
        if source_img is None: return

        kernel_shape = self.kernel_combo.currentText()
        kernel_size = self.default_kernel_size 
        
        try:
            kwargs = {}
            if kernel_shape == 'rectangle':
                kwargs['width'] = kernel_size
            self.result_image = operation_func(source_img, shape=kernel_shape, size=kernel_size, **kwargs)
            self.display_image(self.img_label_result.img_label, self.result_image)
        except Exception as e:
            print(f"Error in morphological operation {operation_func.__name__} with kernel {kernel_shape}: {e}")

    def apply_dilation(self):
        self._apply_morph_op(image_utils.dilate)

    def apply_erosion(self):
        self._apply_morph_op(image_utils.erode)

    def apply_opening(self):
        self._apply_morph_op(image_utils.open_morph)

    def apply_closing(self):
        self._apply_morph_op(image_utils.close_morph)

    def save_result_image(self):
        if self.result_image is None:
            print("No result image to save.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Result Image", "", "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;Bitmap Image (*.bmp)")
        if file_path:
            success = image_utils.save_image(file_path, self.result_image)
            if success:
                print(f"Result image saved to {file_path}")
            else:
                print(f"Failed to save result image to {file_path}")

    def remove_noise_effects(self):
        if self.original_image is None:
            return

        self.noise_image = None
        self.display_image(self.img_label_noise.img_label, self.noise_image)

        # Uncheck any active noise radio button
        checked_noise_button = self.noise_group.checkedButton()
        if checked_noise_button:
            checked_noise_button.setAutoExclusive(False)
            checked_noise_button.setChecked(False)
            checked_noise_button.setAutoExclusive(True)

        # Re-apply current color conversion to the original image for the result panel
        if self.radio_default.isChecked():
            self.apply_convert_default(True)
        elif self.radio_gray.isChecked():
            self.apply_convert_gray(True)
        else:
            # Fallback: if somehow no color conversion is active, clear result
            self.result_image = None
            self.display_image(self.img_label_result.img_label, self.result_image)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessingApp()
    window.show()
    sys.exit(app.exec_())
