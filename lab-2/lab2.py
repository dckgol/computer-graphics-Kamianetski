#!/usr/bin/env python3
"""
Single-file Tkinter image processor.

Содержит адаптированные реализации из:
 - segmentation.py (convolve2d, SegmentationProcessor)
 - histogram.py (HistogramProcessor)
 - image_processor.py (ImageProcessor)
и GUI на tkinter.

Запуск:
    python image_processor_tk.py

Зависимости:
    pillow, numpy
    (опционально) matplotlib для красивой гистограммы
"""

import os
import sys
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
import numpy as np

try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False


def convolve2d(image, kernel):
    """Оптимизированная свертка 2D с использованием векторизации numpy."""
    k_height, k_width = kernel.shape
    img_height, img_width = image.shape

    pad_h = k_height // 2
    pad_w = k_width // 2

    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    result = np.zeros_like(image, dtype=np.float32)

    for i in range(k_height):
        for j in range(k_width):
            result += kernel[i, j] * padded[i:i+img_height, j:j+img_width]

    return result

class SegmentationProcessor:
    @staticmethod
    def point_detection(image, threshold=50):
        img_gray = image.convert('L')
        img_array = np.array(img_gray, dtype=np.float32)

        kernel = np.array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]])

        filtered = convolve2d(img_array, kernel)
        result = np.abs(filtered)
        result = np.where(result > threshold, 255, 0).astype(np.uint8)

        return Image.fromarray(result)

    @staticmethod
    def line_detection(image, direction='horizontal', threshold=100):
        img_gray = image.convert('L')
        img_array = np.array(img_gray, dtype=np.float32)

        kernels = {
            'horizontal': np.array([[-1, -1, -1],
                                   [ 2,  2,  2],
                                   [-1, -1, -1]]),
            'vertical': np.array([[-1, 2, -1],
                                 [-1, 2, -1],
                                 [-1, 2, -1]]),
            'diagonal_45': np.array([[-1, -1, 2],
                                    [-1,  2, -1],
                                    [ 2, -1, -1]]),
            'diagonal_135': np.array([[ 2, -1, -1],
                                     [-1,  2, -1],
                                     [-1, -1,  2]])
        }

        kernel = kernels.get(direction, kernels['horizontal'])
        filtered = convolve2d(img_array, kernel)
        result = np.abs(filtered)
        result = np.where(result > threshold, 255, 0).astype(np.uint8)

        return Image.fromarray(result)

    @staticmethod
    def edge_detection(image, method='sobel', threshold=100):
        img_gray = image.convert('L')
        img_array = np.array(img_gray, dtype=np.float32)

        if method == 'sobel':
            kernel_x = np.array([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]])
            kernel_y = np.array([[-1, -2, -1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]])
        elif method == 'prewitt':
            kernel_x = np.array([[-1, 0, 1],
                                [-1, 0, 1],
                                [-1, 0, 1]])
            kernel_y = np.array([[-1, -1, -1],
                                [ 0,  0,  0],
                                [ 1,  1,  1]])
        elif method == 'roberts':
            kernel_x = np.array([[1, 0],
                                [0, -1]])
            kernel_y = np.array([[0, 1],
                                [-1, 0]])

            gx = convolve2d(img_array, kernel_x)
            gy = convolve2d(img_array, kernel_y)
            magnitude = np.sqrt(gx**2 + gy**2)
            result = np.where(magnitude > threshold, 255, 0).astype(np.uint8)
            return Image.fromarray(result)
        else:
            kernel_x = np.array([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]])
            kernel_y = np.array([[-1, -2, -1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]])

        gx = convolve2d(img_array, kernel_x)
        gy = convolve2d(img_array, kernel_y)
        magnitude = np.sqrt(gx**2 + gy**2)

        result = np.where(magnitude > threshold, 255, 0).astype(np.uint8)

        return Image.fromarray(result)

    @staticmethod
    def combined_edge_detection(image, threshold=100):
        img_gray = image.convert('L')
        img_array = np.array(img_gray, dtype=np.float32)

        kernels = [
            np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]]),
            np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]]),
            np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]]),
            np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]])
        ]

        results = []
        for kernel in kernels:
            filtered = convolve2d(img_array, kernel)
            results.append(np.abs(filtered))

        combined = np.maximum.reduce(results)
        result = np.where(combined > threshold, 255, 0).astype(np.uint8)

        return Image.fromarray(result)

class HistogramProcessor:
    @staticmethod
    def compute_histogram(image, downsample_for_display=False):
        img_array = np.array(image)

        if downsample_for_display and img_array.size > 1000000:
            step = int(np.sqrt(img_array.size / 500000))
            if len(img_array.shape) == 2:
                img_array = img_array[::step, ::step]
            else:
                img_array = img_array[::step, ::step, :]

        if len(img_array.shape) == 2:
            hist, _ = np.histogram(img_array.flatten(), bins=256, range=(0, 256))
            return hist
        else:
            hist_r, _ = np.histogram(img_array[:,:,0].flatten(), bins=256, range=(0, 256))
            hist_g, _ = np.histogram(img_array[:,:,1].flatten(), bins=256, range=(0, 256))
            hist_b, _ = np.histogram(img_array[:,:,2].flatten(), bins=256, range=(0, 256))
            return hist_r, hist_g, hist_b

    @staticmethod
    def equalize_histogram_rgb(image):
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        img_array = np.array(image)

        if len(img_array.shape) == 2:
            equalized = HistogramProcessor._equalize_channel(img_array)
            return Image.fromarray(equalized.astype(np.uint8))

        result = np.zeros_like(img_array)
        for i in range(3):
            result[:,:,i] = HistogramProcessor._equalize_channel(img_array[:,:,i])

        return Image.fromarray(result.astype(np.uint8))

    @staticmethod
    def equalize_histogram_hsv(image):
        """Эквализация только V-компоненты в HSV пространстве."""
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        img_hsv = image.convert('HSV')
        h, s, v = img_hsv.split()

        v_array = np.array(v)
        v_equalized = HistogramProcessor._equalize_channel(v_array)

        v_eq = Image.fromarray(v_equalized.astype(np.uint8))
        result_hsv = Image.merge('HSV', (h, s, v_eq))

        return result_hsv.convert('RGB')

    @staticmethod
    def _equalize_channel(channel):
        hist, _ = np.histogram(channel.flatten(), 256, [0, 256])

        cdf = hist.cumsum()
        cdf_min = cdf[cdf > 0].min()

        cdf_normalized = ((cdf - cdf_min) * 255 / (cdf[-1] - cdf_min)).astype(np.uint8)

        equalized = cdf_normalized[channel.astype(np.uint8)]

        return equalized

    @staticmethod
    def linear_contrast_stretch(image, min_out=0, max_out=255):
        """Линейное контрастирование."""
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        img_array = np.array(image, dtype=np.float32)

        min_in = img_array.min()
        max_in = img_array.max()

        if max_in == min_in:
            return image

        stretched = (img_array - min_in) * (max_out - min_out) / (max_in - min_in) + min_out
        stretched = np.clip(stretched, 0, 255).astype(np.uint8)

        return Image.fromarray(stretched)

class ImageProcessor:
    def __init__(self):
        self.histogram_processor = HistogramProcessor()
        self.segmentation_processor = SegmentationProcessor()

    def process_histogram_equalize_rgb(self, image):
        return self.histogram_processor.equalize_histogram_rgb(image)

    def process_histogram_equalize_hsv(self, image):
        return self.histogram_processor.equalize_histogram_hsv(image)

    def process_linear_contrast(self, image):
        return self.histogram_processor.linear_contrast_stretch(image)

    def process_point_detection(self, image, threshold=50):
        return self.segmentation_processor.point_detection(image, threshold)

    def process_line_detection(self, image, direction='horizontal', threshold=100):
        return self.segmentation_processor.line_detection(image, direction, threshold)

    def process_edge_detection(self, image, method='sobel', threshold=100):
        return self.segmentation_processor.edge_detection(image, method, threshold)

    def process_combined_edges(self, image, threshold=100):
        return self.segmentation_processor.combined_edge_detection(image, threshold)

    def get_histogram(self, image, downsample_for_display=False):
        return self.histogram_processor.compute_histogram(image, downsample_for_display)

# ---------------------------
# GUI
# ---------------------------

class ImageProcessorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Processor (single-file Tkinter)")
        self.geometry("1100x700")
        self.processor = ImageProcessor()

        self.orig_image = None      # PIL Image
        self.processed_image = None # PIL Image
        self.orig_photo = None      # PhotoImage
        self.proc_photo = None

        self._build_ui()
        self._set_default_controls()

    def _build_ui(self):
        # Frames
        control_frame = ttk.Frame(self, padding=8)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        images_frame = ttk.Frame(self, padding=4)
        images_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(images_frame)
        right_frame = ttk.Frame(images_frame)

        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Controls
        load_btn = ttk.Button(control_frame, text="Загрузить", command=self.load_image)
        load_btn.pack(side=tk.LEFT, padx=4)

        save_btn = ttk.Button(control_frame, text="Сохранить результат", command=self.save_processed)
        save_btn.pack(side=tk.LEFT, padx=4)

        ttk.Label(control_frame, text="Операция:").pack(side=tk.LEFT, padx=(12,4))
        self.operation_cb = ttk.Combobox(control_frame, values=[
            "equalize_rgb",
            "equalize_hsv",
            "linear_contrast",
            "point_detection",
            "line_detection",
            "edge_detection",
            "combined_edges"
        ], state="readonly", width=18)
        self.operation_cb.pack(side=tk.LEFT)
        self.operation_cb.bind("<<ComboboxSelected>>", lambda e: self._update_param_widgets())

        ttk.Button(control_frame, text="Применить", command=self.apply_operation).pack(side=tk.LEFT, padx=8)

        ttk.Button(control_frame, text="Показать гистограмму", command=self.show_histogram).pack(side=tk.LEFT, padx=4)

        # Parameter widgets
        params_frame = ttk.Frame(self, padding=4)
        params_frame.pack(side=tk.TOP, fill=tk.X)

        self.threshold_label = ttk.Label(params_frame, text="Threshold:")
        self.threshold_scale = ttk.Scale(params_frame, from_=0, to=255, orient=tk.HORIZONTAL)
        self.threshold_value = tk.IntVar(value=100)
        self.threshold_scale.config(command=lambda v: self.threshold_value.set(int(float(v))))
        self.threshold_entry = ttk.Entry(params_frame, width=5, textvariable=self.threshold_value)

        self.direction_label = ttk.Label(params_frame, text="Direction:")
        self.direction_cb = ttk.Combobox(params_frame, values=["horizontal","vertical","diagonal_45","diagonal_135"], state="readonly", width=12)

        self.method_label = ttk.Label(params_frame, text="Method:")
        self.method_cb = ttk.Combobox(params_frame, values=["sobel","prewitt","roberts"], state="readonly", width=12)

        # Pack param widgets initially hidden; pack in _update_param_widgets
        # Image canvases
        self.left_canvas = tk.Canvas(left_frame, bg="#222", highlightthickness=0)
        self.right_canvas = tk.Canvas(right_frame, bg="#222", highlightthickness=0)

        self.left_canvas.pack(fill=tk.BOTH, expand=True)
        self.right_canvas.pack(fill=tk.BOTH, expand=True)

        # Status bar
        self.status_var = tk.StringVar(value="Готово")
        status = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status.pack(side=tk.BOTTOM, fill=tk.X)

    def _set_default_controls(self):
        self.operation_cb.set("equalize_rgb")
        self.direction_cb.set("horizontal")
        self.method_cb.set("sobel")
        self.threshold_scale.set(100)
        self._update_param_widgets()

    def _update_param_widgets(self):
        # Сначала убираем все
        for w in (self.threshold_label, self.threshold_scale, self.threshold_entry,
                  self.direction_label, self.direction_cb,
                  self.method_label, self.method_cb):
            w.pack_forget()

        op = self.operation_cb.get()
        params_frame = self.threshold_label.master  # общий frame
        if op in ("point_detection", "line_detection", "edge_detection", "combined_edges"):
            # Threshold нужен
            self.threshold_label.pack(side=tk.LEFT, padx=(6,2))
            self.threshold_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
            self.threshold_entry.pack(side=tk.LEFT, padx=4)
        if op == "line_detection":
            self.direction_label.pack(side=tk.LEFT, padx=(8,2))
            self.direction_cb.pack(side=tk.LEFT, padx=2)
        if op == "edge_detection":
            self.method_label.pack(side=tk.LEFT, padx=(8,2))
            self.method_cb.pack(side=tk.LEFT, padx=2)

    def load_image(self):
        path = filedialog.askopenfilename(title="Выберите изображение",
                                          filetypes=[("Image files","*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"), ("All files","*.*")])
        if not path:
            return
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть изображение:\n{e}")
            return
        self.orig_image = img
        self.processed_image = None
        self._draw_images()
        self.status_var.set(f"Загружено: {os.path.basename(path)}")

    def _draw_images(self):
        # рисуем оригинал и результат, масштабируя по размеру canvas
        if self.orig_image:
            self.orig_photo = self._pil_to_photo_for_canvas(self.orig_image, self.left_canvas)
            self.left_canvas.delete("all")
            self.left_canvas.create_image(0,0, anchor=tk.NW, image=self.orig_photo)
        else:
            self.left_canvas.delete("all")
        if self.processed_image:
            self.proc_photo = self._pil_to_photo_for_canvas(self.processed_image, self.right_canvas)
            self.right_canvas.delete("all")
            self.right_canvas.create_image(0,0, anchor=tk.NW, image=self.proc_photo)
        else:
            self.right_canvas.delete("all")

    def _pil_to_photo_for_canvas(self, pil_img, canvas):
        # Подгоняем изображение под размер canvas, сохраняя аспект
        canvas.update_idletasks()
        w = canvas.winfo_width() or 400
        h = canvas.winfo_height() or 300
        img_w, img_h = pil_img.size
        scale = min(w / img_w, h / img_h, 1.0)
        new_w = max(1, int(img_w * scale))
        new_h = max(1, int(img_h * scale))
        resized = pil_img.resize((new_w, new_h), Image.LANCZOS)
        return ImageTk.PhotoImage(resized)

    def apply_operation(self):
        if self.orig_image is None:
            messagebox.showwarning("Нет изображения", "Сначала загрузите изображение.")
            return

        op = self.operation_cb.get()
        try:
            if op == "equalize_rgb":
                out = self.processor.process_histogram_equalize_rgb(self.orig_image)
            elif op == "equalize_hsv":
                out = self.processor.process_histogram_equalize_hsv(self.orig_image)
            elif op == "linear_contrast":
                out = self.processor.process_linear_contrast(self.orig_image)
            elif op == "point_detection":
                th = int(self.threshold_value.get())
                out = self.processor.process_point_detection(self.orig_image, threshold=th)
            elif op == "line_detection":
                th = int(self.threshold_value.get())
                dirn = self.direction_cb.get()
                out = self.processor.process_line_detection(self.orig_image, direction=dirn, threshold=th)
            elif op == "edge_detection":
                th = int(self.threshold_value.get())
                method = self.method_cb.get()
                out = self.processor.process_edge_detection(self.orig_image, method=method, threshold=th)
            elif op == "combined_edges":
                th = int(self.threshold_value.get())
                out = self.processor.process_combined_edges(self.orig_image, threshold=th)
            else:
                messagebox.showerror("Ошибка", f"Неизвестная операция: {op}")
                return
        except Exception as e:
            messagebox.showerror("Ошибка при обработке", str(e))
            return

        # Результат может быть в градациях серого; переводим в RGB для отображения
        if out.mode == "L":
            out = out.convert("RGB")
        self.processed_image = out
        self._draw_images()
        self.status_var.set(f"Операция '{op}' выполнена")

    def save_processed(self):
        if self.processed_image is None:
            messagebox.showwarning("Нет результата", "Сначала примените операцию.")
            return
        path = filedialog.asksaveasfilename(title="Сохранить результат", defaultextension=".png",
                                            filetypes=[("PNG","*.png"),("JPEG","*.jpg;*.jpeg"),("BMP","*.bmp"),("All files","*.*")])
        if not path:
            return
        try:
            self.processed_image.save(path)
            self.status_var.set(f"Сохранено: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Ошибка сохранения", str(e))

    def show_histogram(self):
        if self.orig_image is None:
            messagebox.showwarning("Нет изображения", "Сначала загрузите изображение.")
            return

        hist = self.processor.get_histogram(self.orig_image, downsample_for_display=True)
        if HAS_MATPLOTLIB:
            self._show_histogram_matplotlib(hist)
        else:
            self._show_histogram_canvas(hist)

    def _show_histogram_matplotlib(self, hist):
        win = tk.Toplevel(self)
        win.title("Гистограмма")
        if isinstance(hist, tuple):
            fig = Figure(figsize=(6,3))
            ax = fig.add_subplot(111)
            ax.plot(hist[0], label='R')
            ax.plot(hist[1], label='G')
            ax.plot(hist[2], label='B')
            ax.set_xlim(0,255)
            ax.legend()
            ax.set_title("Histogram (RGB)")
        else:
            fig = Figure(figsize=(6,3))
            ax = fig.add_subplot(111)
            ax.plot(hist, label='Intensity')
            ax.set_xlim(0,255)
            ax.set_title("Histogram (grayscale)")

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _show_histogram_canvas(self, hist):
        win = tk.Toplevel(self)
        win.title("Гистограмма (упрощённая)")

        c = tk.Canvas(win, width=600, height=300, bg="white")
        c.pack(fill=tk.BOTH, expand=True)

        w = 600
        h = 300
        bins = 256
        margin = 10
        plot_w = w - 2*margin
        plot_h = h - 2*margin

        if isinstance(hist, tuple):
            # Найдём максимальное значение по всем каналам
            maxv = max([h.max() for h in hist])
            colors = ("#ff0000", "#00aa00", "#0000ff")
            for idx, channel in enumerate(hist):
                last_x = None
                last_y = None
                for i, val in enumerate(channel):
                    x = margin + (i / (bins-1)) * plot_w
                    y = margin + plot_h - (val / maxv) * plot_h if maxv>0 else margin + plot_h
                    if last_x is not None:
                        c.create_line(last_x, last_y, x, y, fill=colors[idx])
                    last_x = x; last_y = y
        else:
            maxv = hist.max()
            last_x = None; last_y = None
            for i, val in enumerate(hist):
                x = margin + (i / (bins-1)) * plot_w
                y = margin + plot_h - (val / maxv) * plot_h if maxv>0 else margin + plot_h
                if last_x is not None:
                    c.create_line(last_x, last_y, x, y, fill="#000000")
                last_x = x; last_y = y

# ---------------------------
# Запуск
# ---------------------------
def main():
    app = ImageProcessorApp()
    app.mainloop()

if __name__ == "__main__":
    main()
