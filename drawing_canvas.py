import math
import time

import cv2
import numpy as np


class LowPassFilter:
    def __init__(self):
        self.initialized = False
        self.prev = 0.0

    def apply(self, value, alpha):
        if not self.initialized:
            self.prev = value
            self.initialized = True
            return value

        self.prev = alpha * value + (1.0 - alpha) * self.prev
        return self.prev

    def reset(self):
        self.initialized = False
        self.prev = 0.0


class OneEuroFilter1D:
    def __init__(self, min_cutoff=1.8, beta=0.08, d_cutoff=1.2):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.last_time = None
        self.last_raw = None

    @staticmethod
    def _alpha(dt, cutoff):
        cutoff = max(1e-6, cutoff)
        dt = max(1e-6, dt)
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, value, timestamp=None):
        if timestamp is None:
            timestamp = time.perf_counter()

        if self.last_time is None:
            self.last_time = timestamp
            self.last_raw = value
            self.x_filter.reset()
            self.dx_filter.reset()
            return self.x_filter.apply(value, 1.0)

        dt = max(1e-6, timestamp - self.last_time)
        dx = (value - self.last_raw) / dt
        edx = self.dx_filter.apply(dx, self._alpha(dt, self.d_cutoff))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        filtered = self.x_filter.apply(value, self._alpha(dt, cutoff))

        self.last_time = timestamp
        self.last_raw = value
        return filtered

    def reset(self):
        self.x_filter.reset()
        self.dx_filter.reset()
        self.last_time = None
        self.last_raw = None


class DrawingCanvas:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.highlight_layer = np.zeros((height, width, 3), dtype=np.uint8)

        self.current_tool = "PEN"
        self.current_color = (255, 0, 0)  # blue in BGR

        self.brush_thickness = 7
        self.highlighter_thickness = 32
        self.highlighter_alpha = 0.32
        self.eraser_padding = 18

        self.prev_x = None
        self.prev_y = None
        self.prev_eraser = None

        self.x_filter = OneEuroFilter1D(min_cutoff=1.8, beta=0.08, d_cutoff=1.2)
        self.y_filter = OneEuroFilter1D(min_cutoff=1.8, beta=0.08, d_cutoff=1.2)

        self.min_draw_distance = 1.0

    def reset_pointer(self):
        self.prev_x = None
        self.prev_y = None
        self.prev_eraser = None
        self.x_filter.reset()
        self.y_filter.reset()

    def clear(self):
        self.canvas[:] = 0
        self.highlight_layer[:] = 0
        self.reset_pointer()

    def set_tool_from_gesture(self, gesture):
        if gesture == "SELECT_BLUE":
            self.current_tool = "PEN"
            self.current_color = (255, 0, 0)
        elif gesture == "SELECT_GREEN":
            self.current_tool = "PEN"
            self.current_color = (0, 255, 0)
        elif gesture == "SELECT_HIGHLIGHTER":
            self.current_tool = "HIGHLIGHTER"
            self.current_color = (0, 255, 255)
        elif gesture == "SELECT_ERASER":
            self.current_tool = "ERASER"

    def get_smoothed_point(self, x, y, timestamp=None):
        sx = int(self.x_filter.filter(float(x), timestamp))
        sy = int(self.y_filter.filter(float(y), timestamp))
        return sx, sy

    def _draw_segment(self, layer, x1, y1, x2, y2, color, thickness):
        dist = math.hypot(x2 - x1, y2 - y1)
        steps = max(1, int(dist / 2.0))

        prev_ix, prev_iy = x1, y1
        for i in range(1, steps + 1):
            t = i / steps
            ix = int(x1 + (x2 - x1) * t)
            iy = int(y1 + (y2 - y1) * t)
            cv2.line(
                layer,
                (prev_ix, prev_iy),
                (ix, iy),
                color,
                thickness,
                cv2.LINE_AA,
            )
            prev_ix, prev_iy = ix, iy

    def draw_action(self, x, y, timestamp=None):
        x, y = self.get_smoothed_point(x, y, timestamp)

        if self.prev_x is None or self.prev_y is None:
            self.prev_x, self.prev_y = x, y
            return (x, y)

        dist = math.hypot(x - self.prev_x, y - self.prev_y)
        if dist < self.min_draw_distance:
            return (x, y)

        if self.current_tool == "PEN":
            self._draw_segment(
                self.canvas,
                self.prev_x,
                self.prev_y,
                x,
                y,
                self.current_color,
                self.brush_thickness,
            )
        elif self.current_tool == "HIGHLIGHTER":
            self._draw_segment(
                self.highlight_layer,
                self.prev_x,
                self.prev_y,
                x,
                y,
                self.current_color,
                self.highlighter_thickness,
            )

        self.prev_x, self.prev_y = x, y
        return (x, y)

    def _erase_rect(self, rect):
        x1, y1, x2, y2 = rect
        x1 = max(0, min(self.width - 1, x1))
        y1 = max(0, min(self.height - 1, y1))
        x2 = max(0, min(self.width - 1, x2))
        y2 = max(0, min(self.height - 1, y2))

        cv2.rectangle(self.canvas, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.rectangle(self.highlight_layer, (x1, y1), (x2, y2), (0, 0, 0), -1)

    def erase_with_hand_bbox(self, bbox):
        x1, y1, x2, y2 = bbox
        x1 -= self.eraser_padding
        y1 -= self.eraser_padding
        x2 += self.eraser_padding
        y2 += self.eraser_padding

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1

        current = (cx, cy, w, h)

        if self.prev_eraser is None:
            self.prev_eraser = current
            self._erase_rect((int(x1), int(y1), int(x2), int(y2)))
            return

        pcx, pcy, pw, ph = self.prev_eraser
        dist = math.hypot(cx - pcx, cy - pcy)
        steps = max(1, int(dist / 6.0))

        for i in range(1, steps + 1):
            t = i / steps
            icx = pcx + (cx - pcx) * t
            icy = pcy + (cy - pcy) * t
            iw = pw + (w - pw) * t
            ih = ph + (h - ph) * t

            rx1 = int(icx - iw / 2.0)
            ry1 = int(icy - ih / 2.0)
            rx2 = int(icx + iw / 2.0)
            ry2 = int(icy + ih / 2.0)
            self._erase_rect((rx1, ry1, rx2, ry2))

        self.prev_eraser = current

    def overlay_on_frame(self, frame):
        output = frame.copy()

        pen_mask = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, pen_mask = cv2.threshold(pen_mask, 8, 255, cv2.THRESH_BINARY)
        pen_mask_inv = cv2.bitwise_not(pen_mask)

        bg = cv2.bitwise_and(output, output, mask=pen_mask_inv)
        fg = cv2.bitwise_and(self.canvas, self.canvas, mask=pen_mask)
        output = cv2.add(bg, fg)

        highlight_mask = cv2.cvtColor(self.highlight_layer, cv2.COLOR_BGR2GRAY)
        _, highlight_mask = cv2.threshold(highlight_mask, 8, 255, cv2.THRESH_BINARY)
        if np.any(highlight_mask):
            highlight_fg = cv2.bitwise_and(
                self.highlight_layer,
                self.highlight_layer,
                mask=highlight_mask,
            )
            output = cv2.addWeighted(output, 1.0, highlight_fg, self.highlighter_alpha, 0)

        return output