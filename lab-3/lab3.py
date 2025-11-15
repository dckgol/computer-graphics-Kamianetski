#!/usr/bin/env python3
# rasterization_tkinter_full.py
# Единый файл Tkinter: растризация с координатной сеткой, масштабом,
# цветами алгоритмов и возможностью ввода координат начала/конца.
# Обновлено: улучшены реализации Wu и Castle-Pitteway, накопление примитивов.

import tkinter as tk
from tkinter import ttk
import math
import time

# -----------------------------
# Rasterization algorithms
# -----------------------------
class RasterizationAlgorithms:
    @staticmethod
    def step_by_step(x0, y0, x1, y1):
        pixels = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        if dx == 0 and dy == 0:
            return [(x0, y0)]
        steps = max(dx, dy)
        x_step = (x1 - x0) / steps
        y_step = (y1 - y0) / steps
        for i in range(steps + 1):
            x = round(x0 + i * x_step)
            y = round(y0 + i * y_step)
            pixels.append((x, y))
        return pixels

    @staticmethod
    def dda(x0, y0, x1, y1):
        # symmetric DDA: steps = max(|dx|,|dy|)
        pixels = []
        dx = x1 - x0
        dy = y1 - y0
        steps = max(abs(dx), abs(dy))
        if steps == 0:
            return [(x0, y0)]
        x_inc = dx / steps
        y_inc = dy / steps
        x = x0
        y = y0
        for _ in range(steps + 1):
            pixels.append((round(x), round(y)))
            x += x_inc
            y += y_inc
        return pixels

    @staticmethod
    def bresenham_line(x0, y0, x1, y1):
        x0 = int(x0); y0 = int(y0); x1 = int(x1); y1 = int(y1)
        pixels = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x1 >= x0 else -1
        sy = 1 if y1 >= y0 else -1

        if dx >= dy:
            # iterate x
            err = 2*dy - dx  # E' = 2*Dy - Dx
            x = x0; y = y0
            for _ in range(dx + 1):
                pixels.append((x, y))
                if err >= 0:
                    y += sy
                    err += 2*(dy - dx)
                else:
                    err += 2*dy
                x += sx
        else:
            # iterate y
            err = 2*dx - dy
            x = x0; y = y0
            for _ in range(dy + 1):
                pixels.append((x, y))
                if err >= 0:
                    x += sx
                    err += 2*(dx - dy)
                else:
                    err += 2*dx
                y += sy
        return pixels

    @staticmethod
    def bresenham_circle(xc, yc, r):
        # Integer Bresenham circle algorithm (as in presentation)
        pixels = []
        x = 0
        y = int(r)
        d = 3 - 2 * r
        while x <= y:
            # 8 octants
            pixels.append((xc + x, yc + y))
            pixels.append((xc - x, yc + y))
            pixels.append((xc + x, yc - y))
            pixels.append((xc - x, yc - y))
            pixels.append((xc + y, yc + x))
            pixels.append((xc - y, yc + x))
            pixels.append((xc + y, yc - x))
            pixels.append((xc - y, yc - x))
            if d >= 0:
                d = d + 4*(x - y) + 10
                y -= 1
            else:
                d = d + 4*x + 6
            x += 1
        # unique
        return list(dict.fromkeys(pixels))

    @staticmethod
    def wu_line(x0, y0, x1, y1):
        # Xiaolin Wu's line algorithm. Returns list of (x,y,intensity) with intensity in [0,1].
        def ipart(x): return int(math.floor(x))
        def roundi(x): return int(math.floor(x + 0.5))
        def fpart(x): return x - math.floor(x)
        def rfpart(x): return 1 - fpart(x)

        pixels = []
        # work in floats
        x0f = float(x0); y0f = float(y0); x1f = float(x1); y1f = float(y1)
        steep = abs(y1f - y0f) > abs(x1f - x0f)
        if steep:
            x0f, y0f = y0f, x0f
            x1f, y1f = y1f, x1f
        if x0f > x1f:
            x0f, x1f = x1f, x0f
            y0f, y1f = y1f, y0f

        dx = x1f - x0f
        dy = y1f - y0f
        gradient = dy / dx if dx != 0.0 else 0.0

        # handle first endpoint
        xend = roundi(x0f)
        yend = y0f + gradient * (xend - x0f)
        xgap = rfpart(x0f + 0.5)
        xpxl1 = xend
        ypxl1 = ipart(yend)
        if steep:
            pixels.append((ypxl1,   xpxl1, rfpart(yend) * xgap))
            pixels.append((ypxl1+1, xpxl1, fpart(yend)  * xgap))
        else:
            pixels.append((xpxl1, ypxl1,   rfpart(yend) * xgap))
            pixels.append((xpxl1, ypxl1+1, fpart(yend)  * xgap))
        intery = yend + gradient

        # handle second endpoint
        xend = roundi(x1f)
        yend = y1f + gradient * (xend - x1f)
        xgap = fpart(x1f + 0.5)
        xpxl2 = xend
        ypxl2 = ipart(yend)

        # main loop
        for x in range(xpxl1 + 1, xpxl2):
            yflo = intery
            if steep:
                pixels.append((ipart(yflo), x, rfpart(yflo)))
                pixels.append((ipart(yflo)+1, x, fpart(yflo)))
            else:
                pixels.append((x, ipart(yflo), rfpart(yflo)))
                pixels.append((x, ipart(yflo)+1, fpart(yflo)))
            intery = intery + gradient

        # second endpoint
        if steep:
            pixels.append((ypxl2,   xpxl2, rfpart(yend) * xgap))
            pixels.append((ypxl2+1, xpxl2, fpart(yend)  * xgap))
        else:
            pixels.append((xpxl2, ypxl2,   rfpart(yend) * xgap))
            pixels.append((xpxl2, ypxl2+1, fpart(yend)  * xgap))

        # merge duplicates taking max intensity
        merged = {}
        for x, y, w in pixels:
            key = (int(x), int(y))
            merged[key] = max(merged.get(key, 0.0), float(w))
        return [(x, y, merged[(x, y)]) for x, y in merged]

    @staticmethod
    def castle_pitteway(x0, y0, x1, y1):
        # Improved Castle-Pitteway (integer) approximation.
        # Implementation produces main-pixel with intensity based on distance to ideal line and
        # assigns residual intensity to adjacent pixel. This yields a similar visual effect
        # to the algorithm described in lecture slides while being robust.
        x0 = int(x0); y0 = int(y0); x1 = int(x1); y1 = int(y1)
        dx = x1 - x0
        dy = y1 - y0
        if dx == 0 and dy == 0:
            return [(x0, y0, 1.0)]
        steep = abs(dy) > abs(dx)
        # transform to first octant
        if steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1
            dx, dy = x1 - x0, y1 - y0
        sx = 1 if dx >= 0 else -1
        sy = 1 if dy >= 0 else -1
        dx = abs(dx); dy = abs(dy)

        pixels = []
        if dx == 0:
            # vertical after transform
            for i in range(0, dy+1):
                gx = x0
                gy = y0 + i*sy
                if steep:
                    pixels.append((gy, gx, 1.0))
                else:
                    pixels.append((gx, gy, 1.0))
            return pixels

        gradient = dy / dx
        y = y0
        for x in range(x0, x1 + (1 if sx>0 else -1), sx):
            ideal_y = y0 + gradient * (x - x0)
            y_near = int(round(ideal_y))
            dist = abs(ideal_y - y_near)
            intensity_near = max(0.0, 1.0 - dist)
            intensity_far = max(0.0, dist)
            if steep:
                # swap back
                pixels.append((y_near, x, intensity_near))
                if intensity_far > 1e-6:
                    # choose neighbor towards sign of gradient
                    neighbor = y_near + (1 if (ideal_y - y_near) > 0 else -1)
                    pixels.append((neighbor, x, intensity_far))
            else:
                pixels.append((x, y_near, intensity_near))
                if intensity_far > 1e-6:
                    neighbor = y_near + (1 if (ideal_y - y_near) > 0 else -1)
                    pixels.append((x, neighbor, intensity_far))
        # merge
        merged = {}
        for x, y, w in pixels:
            k = (int(x), int(y))
            merged[k] = max(merged.get(k, 0.0), float(w))
        return [(x, y, merged[(x, y)]) for x, y in merged]

# -----------------------------
# Performance tester
# -----------------------------
class PerformanceTester:
    @staticmethod
    def measure_time(func, iterations=200):
        start = time.perf_counter()
        for _ in range(iterations):
            func()
        end = time.perf_counter()
        return (end - start) / iterations * 1_000_000

    @staticmethod
    def benchmark_all():
        tests = [
            ("Step-by-step", lambda: RasterizationAlgorithms.step_by_step(0, 0, 150, 100)),
            ("DDA", lambda: RasterizationAlgorithms.dda(0, 0, 150, 100)),
            ("Bresenham", lambda: RasterizationAlgorithms.bresenham_line(0, 0, 150, 100)),
            ("Wu", lambda: RasterizationAlgorithms.wu_line(0, 0, 150, 100)),
            ("Castle-Pitteway", lambda: RasterizationAlgorithms.castle_pitteway(0, 0, 150, 100)),
            ("Bresenham circle", lambda: RasterizationAlgorithms.bresenham_circle(0, 0, 80)),
        ]
        results = []
        for name, fn in tests:
            t = PerformanceTester.measure_time(fn, iterations=300)
            pixels = fn()
            results.append((name, t, len(pixels)))
        return results

# -----------------------------
# Helper color functions
# -----------------------------
def hex_to_rgb(hexcol):
    hexcol = hexcol.lstrip('#')
    return tuple(int(hexcol[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb

def blend_with_white(base_hex, weight):
    # weight 0..1 — 1 means full base color, 0 means white
    br, bg, bb = hex_to_rgb(base_hex)
    wr, wg, wb = 255, 255, 255
    r = int(round(wr * (1 - weight) + br * weight))
    g = int(round(wg * (1 - weight) + bg * weight))
    b = int(round(wb * (1 - weight) + bb * weight))
    return rgb_to_hex((r, g, b))

# -----------------------------
# Main Tkinter app
# -----------------------------
class RasterizationApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Rasterization with Colors and Coordinate Input")
        self.canvas_w = 900
        self.canvas_h = 650
        self.pixel_size = 6

        # colors per algorithm
        self.colors = {
            "Step-by-step": "#d62728",
            "DDA": "#2ca02c",
            "Bresenham": "#1f77b4",
            "Wu": "#9467bd",
            "Castle-Pitteway": "#ff7f0e",
            "Circle": "#17becf",
        }

        # store drawn shapes so they accumulate
        # each item: dict { 'pixels': [(x,y[,w]),...], 'color': '#rrggbb', 'algo': name }
        self.shapes = []

        self.start_pt = None
        self.end_pt = None
        self._build_ui()

    def _build_ui(self):
        control = ttk.Frame(self)
        control.pack(fill=tk.X, pady=5)

        ttk.Label(control, text="Algorithm:").pack(side=tk.LEFT)
        self.algo_var = tk.StringVar(value="Bresenham")
        self.algo_box = ttk.Combobox(control, textvariable=self.algo_var,
                                     values=["Step-by-step", "DDA", "Bresenham",
                                             "Wu", "Castle-Pitteway", "Circle"],
                                     width=18, state="readonly")
        self.algo_box.pack(side=tk.LEFT, padx=5)

        # coordinate inputs
        ttk.Label(control, text="Start x,y:").pack(side=tk.LEFT, padx=5)
        self.start_x = tk.Entry(control, width=4)
        self.start_y = tk.Entry(control, width=4)
        self.start_x.pack(side=tk.LEFT); self.start_y.pack(side=tk.LEFT)
        self.start_x.insert(0, "0"); self.start_y.insert(0, "0")

        ttk.Label(control, text="End x,y:").pack(side=tk.LEFT, padx=5)
        self.end_x = tk.Entry(control, width=4)
        self.end_y = tk.Entry(control, width=4)
        self.end_x.pack(side=tk.LEFT); self.end_y.pack(side=tk.LEFT)
        self.end_x.insert(0, "20"); self.end_y.insert(0, "10")

        ttk.Button(control, text="Draw coords", command=self.draw_from_entries).pack(side=tk.LEFT, padx=6)

        # scale
        ttk.Label(control, text="Scale(px/cell):").pack(side=tk.LEFT, padx=5)
        self.scale_var = tk.StringVar(value=str(self.pixel_size))
        self.scale_entry = ttk.Entry(control, width=4, textvariable=self.scale_var)
        self.scale_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(control, text="Apply", command=self.apply_scale).pack(side=tk.LEFT, padx=5)

        ttk.Button(control, text="Clear (keep grid)", command=self.clear_canvas).pack(side=tk.LEFT, padx=5)
        ttk.Button(control, text="Clear All", command=self.clear_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(control, text="Benchmark", command=self.run_benchmark).pack(side=tk.LEFT, padx=5)

        self.canvas = tk.Canvas(self, width=self.canvas_w, height=self.canvas_h, bg="white")
        self.canvas.pack(pady=5)
        self.canvas.bind("<Button-1>", self.on_click)

        # info / legend
        bottom = ttk.Frame(self)
        bottom.pack(fill=tk.BOTH, expand=True)
        self.info = tk.Text(bottom, height=8)
        self.info.pack(fill=tk.BOTH, expand=True)
        self._draw_legend()
        self.draw_grid_and_axes()

    # UI helpers
    def _draw_legend(self):
        self.info.delete('1.0', tk.END)
        self.info.insert(tk.END, "Algorithm colors:")
        for name, col in self.colors.items():
            self.info.insert(tk.END, f"  {name}: {col}")
        self.info.insert(tk.END, "Click on canvas to pick start and end (first click = start, second = end).")
        self.info.insert(tk.END, "Or input coordinates in fields and press Draw coords.")
        self.info.insert(tk.END, "Use 'Clear (keep grid)' to remove drawn pixels but keep grid (or 'Clear All' to reset).")

    # coordinate mapping
    def grid_to_screen(self, gx, gy):
        ps = self.pixel_size
        ox = self.canvas_w // 2
        oy = self.canvas_h // 2
        sx = ox + gx * ps
        sy = oy - gy * ps
        return sx, sy

    def screen_to_grid(self, sx, sy):
        ps = self.pixel_size
        ox = self.canvas_w // 2
        oy = self.canvas_h // 2
        gx = round((sx - ox) / ps)
        gy = round((oy - sy) / ps)
        return gx, gy

    # grid and axes
    def draw_grid_and_axes(self):
        # draw grid under everything using tag 'grid'
        self.canvas.delete('grid')
        ps = self.pixel_size
        w = self.canvas_w; h = self.canvas_h
        ox = w // 2; oy = h // 2

        max_x = w // ps; max_y = h // ps
        for i in range(-max_x, max_x + 1):
            x = ox + i * ps
            if i % 10 == 0:
                color = '#999'; width = 2
            elif i % 5 == 0:
                color = '#ccc'; width = 1
            else:
                color = '#eee'; width = 1
            self.canvas.create_line(x, 0, x, h, fill=color, width=width, tags=('grid'))
            if i % 10 == 0:
                self.canvas.create_text(x + 6, oy + 12, text=str(i), font=('Arial', 8), tags=('grid'))

        for j in range(-max_y, max_y + 1):
            y = oy - j * ps
            if j % 10 == 0:
                color = '#999'; width = 2
            elif j % 5 == 0:
                color = '#ccc'; width = 1
            else:
                color = '#eee'; width = 1
            self.canvas.create_line(0, y, w, y, fill=color, width=width, tags=('grid'))
            if j % 10 == 0:
                self.canvas.create_text(ox + 18, y - 6, text=str(j), font=('Arial', 8), tags=('grid'))

        # axes
        self.canvas.create_line(0, oy, w, oy, fill='black', width=2, tags=('grid'))
        self.canvas.create_line(ox, 0, ox, h, fill='black', width=2, tags=('grid'))
        self.canvas.create_text(ox + 26, oy - 20, text='+X', font=('Arial', 10, 'bold'), tags=('grid'))
        self.canvas.create_text(ox - 20, oy - 30, text='+Y', font=('Arial', 10, 'bold'), tags=('grid'))

    # drawing
    def draw_pixels(self, pixels, base_color=None):
        # draw into tag 'drawn' so we can selectively clear drawn pixels
        ps = self.pixel_size
        for p in pixels:
            if len(p) == 3:
                gx, gy, w = p
            else:
                gx, gy = p
                w = 1.0
            sx, sy = self.grid_to_screen(gx, gy)
            if base_color:
                color = blend_with_white(base_color, w)
            else:
                color_int = int(255 * (1 - w))
                color = f"#{color_int:02x}{color_int:02x}{color_int:02x}"
            # draw cell as rectangle anchored at (sx,sy)
            # store rectangle with tag 'drawn'
            self.canvas.create_rectangle(sx, sy, sx + ps, sy + ps, fill=color, outline=color, tags=('drawn',))

    def redraw_all_shapes(self):
        # clear drawn layer and redraw saved shapes
        self.canvas.delete('drawn')
        # ensure grid is below: redraw grid to keep tag order
        self.draw_grid_and_axes()
        for sh in self.shapes:
            self.draw_pixels(sh['pixels'], base_color=sh['color'])

    def draw_line(self):
        if self.start_pt is None or self.end_pt is None:
            return
        x0, y0 = self.start_pt
        x1, y1 = self.end_pt
        algo = self.algo_var.get()
        base_color = self.colors.get(algo, '#000000')

        if algo == 'Step-by-step':
            pts = RasterizationAlgorithms.step_by_step(x0, y0, x1, y1)
            pts = [(x, y, 1.0) for x, y in pts]
        elif algo == 'DDA':
            pts = RasterizationAlgorithms.dda(x0, y0, x1, y1)
            pts = [(x, y, 1.0) for x, y in pts]
        elif algo == 'Bresenham':
            pts = RasterizationAlgorithms.bresenham_line(x0, y0, x1, y1)
            pts = [(x, y, 1.0) for x, y in pts]
        elif algo == 'Wu':
            pts = RasterizationAlgorithms.wu_line(x0, y0, x1, y1)
        elif algo == 'Castle-Pitteway':
            pts = RasterizationAlgorithms.castle_pitteway(x0, y0, x1, y1)
        else:  # Circle
            r = int(round(math.hypot(x1 - x0, y1 - y0)))
            pts = RasterizationAlgorithms.bresenham_circle(x0, y0, r)
            pts = [(x, y, 1.0) for x, y in pts]

        # append shape to shapes list so it remains
        self.shapes.append({'pixels': pts, 'color': base_color, 'algo': algo})
        # redraw all shapes
        self.redraw_all_shapes()
        self.info.insert(tk.END, f"Appended {len(pts)} pixels using {algo}")
        self.info.see(tk.END)

    # UI actions
    def draw_from_entries(self):
        try:
            x0 = int(self.start_x.get()); y0 = int(self.start_y.get())
            x1 = int(self.end_x.get()); y1 = int(self.end_y.get())
        except Exception:
            self.info.insert(tk.END, "Invalid integer coordinates")
            return
        self.start_pt = (x0, y0); self.end_pt = (x1, y1)
        self.draw_line()
        # reset stored points for click-mode
        self.start_pt = None; self.end_pt = None

    def apply_scale(self):
        try:
            s = int(self.scale_var.get())
            if s <= 0: raise ValueError
            self.pixel_size = s
            # when scale changes, need to redraw everything
            self.redraw_all_shapes()
        except Exception:
            self.info.insert(tk.END, 'Invalid scale')

    def clear_canvas(self):
        # remove drawn pixels but keep shapes list? Provide two options: here clear drawn only
        self.canvas.delete('drawn')
        # keep shapes list intact but remove visual rectangles; user can use Clear All to reset shapes too
        self.info.insert(tk.END, 'Cleared drawn layer (shapes still stored). Use "Clear All" to remove stored shapes.')

    def clear_all(self):
        self.canvas.delete('all')
        self.shapes.clear()
        self.draw_grid_and_axes()
        self.info.delete('1.0', tk.END)
        self._draw_legend()

    def on_click(self, event):
        gx, gy = self.screen_to_grid(event.x, event.y)
        if self.start_pt is None:
            self.start_pt = (gx, gy)
            # update entries
            self.start_x.delete(0, tk.END); self.start_x.insert(0, str(gx))
            self.start_y.delete(0, tk.END); self.start_y.insert(0, str(gy))
            self.info.insert(tk.END, f"Start set to {self.start_pt}")
        else:
            self.end_pt = (gx, gy)
            self.end_x.delete(0, tk.END); self.end_x.insert(0, str(gx))
            self.end_y.delete(0, tk.END); self.end_y.insert(0, str(gy))
            self.info.insert(tk.END, f"End set to {self.end_pt} — drawing...")
            self.draw_line()
            # reset for next
            self.start_pt = None; self.end_pt = None

    def run_benchmark(self):
        self.info.insert(tk.END, 'Running benchmark...')
        results = PerformanceTester.benchmark_all()
        for name, t, p in results:
            self.info.insert(tk.END, f"  {name}: {t:.2f} us, pixels={p}")
        self.info.see(tk.END)

# -----------------------------
# Run
# -----------------------------
if __name__ == '__main__':
    app = RasterizationApp()
    app.mainloop()