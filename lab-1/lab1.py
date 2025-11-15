import tkinter as tk
from tkinter import ttk, colorchooser

class ColorConverterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Цветовые модели: CMYK, RGB, HSV")
        self.root.geometry("800x600")
        
        self.c_var = tk.DoubleVar(value=0)
        self.m_var = tk.DoubleVar(value=0)
        self.y_var = tk.DoubleVar(value=0)
        self.k_var = tk.DoubleVar(value=0)
        
        self.r_var = tk.DoubleVar(value=0)
        self.g_var = tk.DoubleVar(value=0)
        self.b_var = tk.DoubleVar(value=0)
        
        self.h_var = tk.DoubleVar(value=0)
        self.s_var = tk.DoubleVar(value=0)
        self.v_var = tk.DoubleVar(value=0)
        
        self.updating = False
        
        self.cmyk_scales = []
        self.rgb_scales = []
        self.hsv_scales = []
        
        self.setup_ui()
        self.bind_events()
        
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        self.color_display = tk.Label(main_frame, bg='#000000', width=30, height=3, 
                                     font=("Arial", 12), relief="solid", bd=2)
        self.color_display.grid(row=0, column=0, columnspan=2, pady=(0, 20), sticky=(tk.W, tk.E))
        
        self.hex_var = tk.StringVar(value="#000000")
        hex_label = tk.Label(main_frame, textvariable=self.hex_var, font=("Arial", 12, "bold"))
        hex_label.grid(row=1, column=0, columnspan=2, pady=(0, 10))
        
        ttk.Button(main_frame, text="Выбрать из палитры", 
                  command=self.choose_color_from_palette).grid(row=0, column=2, rowspan=2, padx=(10, 0))
        
        notebook = ttk.Notebook(main_frame)
        notebook.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        
        cmyk_frame = ttk.Frame(notebook, padding="10")
        self.create_cmyk_section(cmyk_frame)
        notebook.add(cmyk_frame, text="CMYK")
        
        rgb_frame = ttk.Frame(notebook, padding="10")
        self.create_rgb_section(rgb_frame)
        notebook.add(rgb_frame, text="RGB")
        
        hsv_frame = ttk.Frame(notebook, padding="10")
        self.create_hsv_section(hsv_frame)
        notebook.add(hsv_frame, text="HSV")
        
        info_frame = ttk.Frame(main_frame, padding="5")
        info_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        info_text = (
            "Измените цвет используя:\n"
            "• Поля ввода для точных значений\n"
            "• Ползунки для плавного изменения\n"
            "• Кнопку 'Выбрать из палитры' для визуального выбора"
        )
        info_label = tk.Label(info_frame, text=info_text, justify=tk.LEFT, 
                             font=("Arial", 9), fg="gray")
        info_label.pack(anchor=tk.W)
        
    def create_cmyk_section(self, parent):
        labels = ['Cyan (C):', 'Magenta (M):', 'Yellow (Y):', 'Black (K):']
        vars = [self.c_var, self.m_var, self.y_var, self.k_var]
        
        for i, (label, var) in enumerate(zip(labels, vars)):
            ttk.Label(parent, text=label).grid(row=i, column=0, sticky=tk.W, pady=5)
            
            entry = ttk.Entry(parent, textvariable=var, width=10)
            entry.grid(row=i, column=1, padx=(5, 10), pady=5, sticky=tk.W)
            
            scale = tk.Scale(parent, from_=0, to=100, variable=var, 
                           orient=tk.HORIZONTAL, length=200, showvalue=False,
                           resolution=1, command=self.on_cmyk_scale)
            scale.grid(row=i, column=2, sticky=(tk.W, tk.E), pady=5)
            self.cmyk_scales.append(scale)
            
        parent.columnconfigure(2, weight=1)
    
    def create_rgb_section(self, parent):
        labels = ['Red (R):', 'Green (G):', 'Blue (B):']
        vars = [self.r_var, self.g_var, self.b_var]
        
        for i, (label, var) in enumerate(zip(labels, vars)):
            ttk.Label(parent, text=label).grid(row=i, column=0, sticky=tk.W, pady=5)
            
            entry = ttk.Entry(parent, textvariable=var, width=10)
            entry.grid(row=i, column=1, padx=(5, 10), pady=5, sticky=tk.W)
            
            scale = tk.Scale(parent, from_=0, to=255, variable=var, 
                           orient=tk.HORIZONTAL, length=200, showvalue=False,
                           resolution=1, command=self.on_rgb_scale)
            scale.grid(row=i, column=2, sticky=(tk.W, tk.E), pady=5)
            self.rgb_scales.append(scale)
            
        parent.columnconfigure(2, weight=1)
    
    def create_hsv_section(self, parent):
        labels = ['Hue (H):', 'Saturation (S):', 'Value (V):']
        vars = [self.h_var, self.s_var, self.v_var]
        ranges = [(0, 359), (0, 100), (0, 100)]
        
        for i, (label, var, range_vals) in enumerate(zip(labels, vars, ranges)):
            ttk.Label(parent, text=label).grid(row=i, column=0, sticky=tk.W, pady=5)
            
            entry = ttk.Entry(parent, textvariable=var, width=10)
            entry.grid(row=i, column=1, padx=(5, 10), pady=5, sticky=tk.W)
            
            scale = tk.Scale(parent, from_=range_vals[0], to=range_vals[1], 
                           variable=var, orient=tk.HORIZONTAL, length=200, 
                           showvalue=False, resolution=1, command=self.on_hsv_scale)
            scale.grid(row=i, column=2, sticky=(tk.W, tk.E), pady=5)
            self.hsv_scales.append(scale)
            
        parent.columnconfigure(2, weight=1)
    
    def bind_events(self):
        for var in [self.c_var, self.m_var, self.y_var, self.k_var]:
            var.trace_add('write', self.on_cmyk_entry)
        
        for var in [self.r_var, self.g_var, self.b_var]:
            var.trace_add('write', self.on_rgb_entry)
        
        for var in [self.h_var, self.s_var, self.v_var]:
            var.trace_add('write', self.on_hsv_entry)
    
    def on_cmyk_scale(self, value):
        if self.updating:
            return
        self.updating = True
        self.update_from_cmyk()
        self.updating = False
    
    def on_rgb_scale(self, value):
        if self.updating:
            return
        self.updating = True
        self.update_from_rgb()
        self.updating = False
    
    def on_hsv_scale(self, value):
        if self.updating:
            return
        self.updating = True
        self.update_from_hsv()
        self.updating = False
    
    def on_cmyk_entry(self, *args):
        if self.updating:
            return
        self.updating = True
        try:
            c = float(self.c_var.get())
            m = float(self.m_var.get())
            y = float(self.y_var.get())
            k = float(self.k_var.get())
            
            self.c_var.set(max(0, min(100, c)))
            self.m_var.set(max(0, min(100, m)))
            self.y_var.set(max(0, min(100, y)))
            self.k_var.set(max(0, min(100, k)))
            
            self.update_from_cmyk()
        except (ValueError, tk.TclError):
            pass
        finally:
            self.updating = False
    
    def on_rgb_entry(self, *args):
        if self.updating:
            return
        self.updating = True
        try:
            r = float(self.r_var.get())
            g = float(self.g_var.get())
            b = float(self.b_var.get())
            
            r = max(0, min(255, r))
            g = max(0, min(255, g))
            b = max(0, min(255, b))
            
            self.r_var.set(r)
            self.g_var.set(g)
            self.b_var.set(b)
            
            self.update_from_rgb()
        except (ValueError, tk.TclError):
            pass
        finally:
            self.updating = False
    
    def on_hsv_entry(self, *args):
        if self.updating:
            return
        self.updating = True
        try:
            h = float(self.h_var.get())
            s = float(self.s_var.get())
            v = float(self.v_var.get())
            
            self.h_var.set(max(0, min(359, h)))
            self.s_var.set(max(0, min(100, s)))
            self.v_var.set(max(0, min(100, v)))
            
            self.update_from_hsv()
        except (ValueError, tk.TclError):
            pass
        finally:
            self.updating = False
    
    def choose_color_from_palette(self):
        color = colorchooser.askcolor(title="Выберите цвет")
        if color[0] is not None:
            r, g, b = [int(c) for c in color[0]]
            if not self.updating:
                self.updating = True
                self.update_from_rgb_values(r, g, b)
                self.updating = False
    
    def update_from_cmyk(self):
        c = self.c_var.get() / 100.0
        m = self.m_var.get() / 100.0
        y = self.y_var.get() / 100.0
        k = self.k_var.get() / 100.0
        
        r, g, b = self.cmyk_to_rgb(c, m, y, k)

        h, s, v = self.rgb_to_hsv(r, g, b)
        self.h_var.set(round(h, 2))
        self.s_var.set(round(s * 100, 2))
        self.v_var.set(round(v * 100, 2))
        
        hex_color = f'#{int(r):02x}{int(g):02x}{int(b):02x}'
        self.color_display.config(bg=hex_color)
        self.hex_var.set(hex_color.upper())
    
    def update_from_rgb(self):
        r = self.r_var.get()
        g = self.g_var.get()
        b = self.b_var.get()
        
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        
        self.r_var.set(r)
        self.g_var.set(g)
        self.b_var.set(b)
        
        self.update_from_rgb_values(r, g, b)
    
    def update_from_hsv(self):
        h = self.h_var.get()
        s = self.s_var.get() / 100.0
        v = self.v_var.get() / 100.0
        
        r, g, b = self.hsv_to_rgb(h, s, v)
        self.update_from_rgb_values(r, g, b)
    
    def update_from_rgb_values(self, r, g, b):
        self.r_var.set(int(r))
        self.g_var.set(int(g))
        self.b_var.set(int(b))
        
        c, m, y, k = self.rgb_to_cmyk(r, g, b)
        self.c_var.set(round(c * 100, 2))
        self.m_var.set(round(m * 100, 2))
        self.y_var.set(round(y * 100, 2))
        self.k_var.set(round(k * 100, 2))
        
        h, s, v = self.rgb_to_hsv(r, g, b)
        self.h_var.set(round(h, 2))
        self.s_var.set(round(s * 100, 2))
        self.v_var.set(round(v * 100, 2))
        
        hex_color = f'#{int(r):02x}{int(g):02x}{int(b):02x}'
        self.color_display.config(bg=hex_color)
        self.hex_var.set(hex_color.upper())
    
    def cmyk_to_rgb(self, c, m, y, k):
        r = 255 * (1 - c) * (1 - k)
        g = 255 * (1 - m) * (1 - k)
        b = 255 * (1 - y) * (1 - k)
        return max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))
    
    def rgb_to_cmyk(self, r, g, b):
        if r == 0 and g == 0 and b == 0:
            return 0, 0, 0, 1
            
        r_norm = r / 255.0
        g_norm = g / 255.0
        b_norm = b / 255.0
        
        k = 1 - max(r_norm, g_norm, b_norm)
        
        if k > 0.9999:
            return 0, 0, 0, 1
        
        c = (1 - r_norm - k) / (1 - k)
        m = (1 - g_norm - k) / (1 - k)
        y = (1 - b_norm - k) / (1 - k)
        
        return c, m, y, k
    
    def rgb_to_hsv(self, r, g, b):
        r_norm = r / 255.0
        g_norm = g / 255.0
        b_norm = b / 255.0
        
        cmax = max(r_norm, g_norm, b_norm)
        cmin = min(r_norm, g_norm, b_norm)
        delta = cmax - cmin
        
        if delta == 0:
            h = 0
        elif cmax == r_norm:
            h = 60 * (((g_norm - b_norm) / delta) % 6)
        elif cmax == g_norm:
            h = 60 * (((b_norm - r_norm) / delta) + 2)
        else:
            h = 60 * (((r_norm - g_norm) / delta) + 4)
        
        s = 0 if cmax == 0 else delta / cmax
        v = cmax
        
        return h, s, v
    
    def hsv_to_rgb(self, h, s, v):
        c = v * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = v - c
        
        if 0 <= h < 60:
            r1, g1, b1 = c, x, 0
        elif 60 <= h < 120:
            r1, g1, b1 = x, c, 0
        elif 120 <= h < 180:
            r1, g1, b1 = 0, c, x
        elif 180 <= h < 240:
            r1, g1, b1 = 0, x, c
        elif 240 <= h < 300:
            r1, g1, b1 = x, 0, c
        else:  # 300 <= h < 360
            r1, g1, b1 = c, 0, x
        
        r = (r1 + m) * 255
        g = (g1 + m) * 255
        b = (b1 + m) * 255
        
        return max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))

if __name__ == "__main__":
    root = tk.Tk()
    app = ColorConverterApp(root)
    root.mainloop()