"""
Drawing utilities: canvas management, line smoothing, toolbar UI, and FPS overlay.

Provides a virtual whiteboard layer composited on top of the webcam feed.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import time
import math
import cv2
import numpy as np


class OneEuroFilter2D:
    """One Euro Filter for 2D noisy signals (like hand tracking).
    
    Effectively eliminates jitter at low speeds (for precise writing) 
    while preserving high responsiveness at high speeds.
    """
    def __init__(self, min_cutoff=0.5, beta=1.5, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.y_prev = None
        self.dx_prev = 0.0
        self.dy_prev = 0.0
        self.t_prev = 0.0

    def __call__(self, x: int, y: int) -> Tuple[int, int]:
        t = time.time()
        if self.x_prev is None:
            self.x_prev, self.y_prev = float(x), float(y)
            self.t_prev = t
            return x, y

        t_e = t - self.t_prev
        if t_e <= 0.0:
            return int(self.x_prev), int(self.y_prev)

        def smoothing_factor(te, cutoff):
            r = 2 * math.pi * cutoff * te
            return r / (r + 1)

        # Filter the derivative (speed)
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx, dy = (x - self.x_prev) / t_e, (y - self.y_prev) / t_e
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        dy_hat = a_d * dy + (1 - a_d) * self.dy_prev

        # Adjust cutoff dynamically based on speed
        speed = math.sqrt(dx_hat**2 + dy_hat**2)
        cutoff = self.min_cutoff + self.beta * speed

        # Filter the signal (position)
        a = smoothing_factor(t_e, cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        y_hat = a * y + (1 - a) * self.y_prev

        self.x_prev, self.y_prev = x_hat, y_hat
        self.dx_prev, self.dy_prev = dx_hat, dy_hat
        self.t_prev = t

        return int(x_hat), int(y_hat)

    def reset(self):
        self.x_prev = None


# --- Layout constants for the modern whiteboard UI ---

TOOLBAR_HEIGHT = 72
TOOLBAR_PADDING = 16
COLOR_SWATCH_SIZE = 36
COLOR_SWATCH_GAP = 12
CLEAR_BUTTON_WIDTH = 100
CLEAR_BUTTON_HEIGHT = 36


@dataclass
class ColorOption:
    """One selectable color in the toolbar."""

    name: str
    bgr: Tuple[int, int, int]
    center: Tuple[int, int] = (0, 0)
    radius: int = COLOR_SWATCH_SIZE // 2


@dataclass
class DrawingCanvas:
    """
    Persistent drawing surface and UI state.

    Strokes are stored on a separate numpy layer and alpha-blended
    over the mirrored webcam frame each frame.
    """

    width: int
    height: int
    brush_size: int = 6

    canvas: np.ndarray = field(init=False)
    selected_color_index: int = 0
    colors: List[ColorOption] = field(default_factory=list)
    clear_button_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)

    _filter: OneEuroFilter2D = field(default_factory=OneEuroFilter2D, init=False)
    _last_draw_point: Optional[Tuple[int, int]] = field(default=None, init=False)
    _hover_index: int = field(default=-1, init=False)
    _clear_hovered: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        # Separate stroke layer — composited over the webcam each frame.
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self._init_colors()
        self._layout_toolbar()

    def _init_colors(self) -> None:
        """Define the palette shown in the top toolbar."""
        palette = [
            ("Coral", (80, 127, 255)),
            ("Sky", (255, 180, 80)),
            ("Mint", (140, 220, 120)),
            ("Sun", (0, 210, 255)),
            ("Lilac", (220, 130, 255)),
            ("White", (255, 255, 255)),
            ("Ink", (40, 40, 40)),
        ]
        self.colors = [ColorOption(name=n, bgr=c) for n, c in palette]

    def _layout_toolbar(self) -> None:
        """Compute pixel positions for color swatches and the clear button."""
        x = TOOLBAR_PADDING + COLOR_SWATCH_SIZE // 2
        y = TOOLBAR_HEIGHT // 2

        for color in self.colors:
            color.center = (x, y)
            color.radius = COLOR_SWATCH_SIZE // 2
            x += COLOR_SWATCH_SIZE + COLOR_SWATCH_GAP

        # Clear button anchored to the right side of the toolbar.
        btn_x2 = self.width - TOOLBAR_PADDING
        btn_x1 = btn_x2 - CLEAR_BUTTON_WIDTH
        btn_y1 = (TOOLBAR_HEIGHT - CLEAR_BUTTON_HEIGHT) // 2
        btn_y2 = btn_y1 + CLEAR_BUTTON_HEIGHT
        self.clear_button_rect = (btn_x1, btn_y1, btn_x2, btn_y2)

    @property
    def current_color(self) -> Tuple[int, int, int]:
        return self.colors[self.selected_color_index].bgr

    def is_in_drawing_area(self, point: Tuple[int, int]) -> bool:
        """Return True if the point is below the toolbar (on the canvas)."""
        _, y = point
        return y >= TOOLBAR_HEIGHT

    def set_toolbar_hover(self, point: Optional[Tuple[int, int]]) -> None:
        """Highlight toolbar control under mouse or fingertip (hover feedback)."""
        self._hover_index = -1
        self._clear_hovered = False
        if point is None:
            return
        x, y = point
        if y > TOOLBAR_HEIGHT:
            return

        x1, y1, x2, y2 = self.clear_button_rect
        if x1 <= x <= x2 and y1 <= y <= y2:
            self._clear_hovered = True
            return

        for idx, color in enumerate(self.colors):
            cx, cy = color.center
            if (x - cx) ** 2 + (y - cy) ** 2 <= (color.radius + 6) ** 2:
                self._hover_index = idx
                break

    def hit_test_toolbar(self, point: Tuple[int, int]) -> Optional[str]:
        """
        Check whether a point is over a toolbar control.

        Returns 'clear' or a color name, or None if nothing was hit.
        Does not clear the canvas — use handle_toolbar_point for that.
        """
        x, y = point
        if y > TOOLBAR_HEIGHT:
            return None

        x1, y1, x2, y2 = self.clear_button_rect
        if x1 <= x <= x2 and y1 <= y <= y2:
            return "clear"

        for idx, color in enumerate(self.colors):
            cx, cy = color.center
            dist_sq = (x - cx) ** 2 + (y - cy) ** 2
            if dist_sq <= (color.radius + 4) ** 2:
                self.selected_color_index = idx
                return color.name

        return None

    def handle_toolbar_point(self, point: Tuple[int, int]) -> Optional[str]:
        """
        Activate a toolbar control from a mouse click or hand pointer.

        Selects color swatches and clears the canvas when Clear is hit.
        """
        hit = self.hit_test_toolbar(point)
        if hit == "clear":
            self.clear()
        return hit

    def clear(self) -> None:
        """Erase all strokes from the canvas."""
        self.canvas[:] = 0
        self._filter.reset()
        self._last_draw_point = None

    def _smooth_point(self, point: Tuple[int, int]) -> Tuple[int, int]:
        """Apply One-Euro Filter to massively reduce jitter for writing."""
        return self._filter(point[0], point[1])

    def add_stroke_point(self, point: Tuple[int, int], is_eraser: bool = False) -> None:
        """
        Add a smoothed stroke segment from the previous point to the new one.
        Uses anti-aliased lines for soft, modern-looking strokes.
        If is_eraser is true, draws black to wipe out strokes.
        """
        if not self.is_in_drawing_area(point):
            return

        smooth = self._smooth_point(point)

        if self._last_draw_point is not None:
            color = (0, 0, 0) if is_eraser else self.current_color
            thickness = self.brush_size * 8 if is_eraser else self.brush_size
            cv2.line(
                self.canvas,
                self._last_draw_point,
                smooth,
                color,
                thickness,
                lineType=cv2.LINE_AA,
            )

        self._last_draw_point = smooth

    def end_stroke(self) -> None:
        """Reset stroke state when the user lifts their drawing gesture."""
        self._filter.reset()
        self._last_draw_point = None

    def draw_toolbar(self, frame: np.ndarray) -> None:
        """Render the modern top toolbar directly onto the output frame."""
        # Frosted dark bar background.
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (0, 0),
            (self.width, TOOLBAR_HEIGHT),
            (45, 42, 38),
            -1,
        )
        cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)

        # Subtle bottom border separating toolbar from canvas.
        cv2.line(
            frame,
            (0, TOOLBAR_HEIGHT - 1),
            (self.width, TOOLBAR_HEIGHT - 1),
            (90, 85, 80),
            1,
            lineType=cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            "Air Draw",
            (self.width // 2 - 50, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (230, 225, 220),
            2,
            lineType=cv2.LINE_AA,
        )

        # Color swatches with selection ring.
        for idx, color in enumerate(self.colors):
            cx, cy = color.center
            cv2.circle(frame, (cx, cy), color.radius, color.bgr, -1, lineType=cv2.LINE_AA)

            highlight = tuple(min(255, c + 40) for c in color.bgr)
            cv2.circle(
                frame,
                (cx - 6, cy - 6),
                max(3, color.radius // 4),
                highlight,
                -1,
                lineType=cv2.LINE_AA,
            )

            if idx == self.selected_color_index:
                cv2.circle(
                    frame,
                    (cx, cy),
                    color.radius + 4,
                    (255, 255, 255),
                    2,
                    lineType=cv2.LINE_AA,
                )
            elif idx == self._hover_index:
                cv2.circle(
                    frame,
                    (cx, cy),
                    color.radius + 3,
                    (200, 200, 210),
                    2,
                    lineType=cv2.LINE_AA,
                )

        # Clear button (mouse click or hand pointer in pause mode).
        x1, y1, x2, y2 = self.clear_button_rect
        fill = (90, 75, 70) if self._clear_hovered else (70, 65, 60)
        border = (180, 140, 130) if self._clear_hovered else (120, 115, 110)
        cv2.rectangle(frame, (x1, y1), (x2, y2), fill, -1, lineType=cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1), (x2, y2), border, 2 if self._clear_hovered else 1, lineType=cv2.LINE_AA)
        cv2.putText(
            frame,
            "Clear",
            (x1 + 18, y2 - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 235, 230),
            1,
            lineType=cv2.LINE_AA,
        )

    def composite(self, frame: np.ndarray) -> np.ndarray:
        """Blend stored strokes onto the live frame."""
        mask = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY) > 0
        frame[mask] = self.canvas[mask]
        return frame

    def draw_status(
        self,
        frame: np.ndarray,
        fps: float,
        mode_label: str,
        hand_detected: bool,
        resolution: str = "",
    ) -> None:
        """Draw FPS counter and current interaction mode in the bottom-left."""
        status_bg = frame.copy()
        status_w = 280 if resolution else 220
        cv2.rectangle(status_bg, (8, self.height - 48), (status_w, self.height - 8), (45, 42, 38), -1)
        cv2.addWeighted(status_bg, 0.75, frame, 0.25, 0, frame)

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (16, self.height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 255, 180),
            1,
            lineType=cv2.LINE_AA,
        )

        hand_text = "Hand: OK" if hand_detected else "Hand: --"
        res_text = f"  |  {resolution}" if resolution else ""
        cv2.putText(
            frame,
            f"{mode_label}  |  {hand_text}{res_text}",
            (16, self.height - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 195, 190),
            1,
            lineType=cv2.LINE_AA,
        )

        # Draw creator credits on the bottom right
        credits = [
            "Created by: Vaibhav Bhardwaj",
            "LinkedIn: www.linkedin.com/in/mr-vaibhav-bhardwaj",
            "GitHub: github.com/mrvaibhavbhardwaj"
        ]
        
        # Calculate bounding box for the credits
        max_width = 380
        credits_h = len(credits) * 20 + 10
        credits_x = self.width - max_width - 10
        credits_y = self.height - credits_h - 10
        
        # Draw translucent background for credits
        credits_bg = frame.copy()
        cv2.rectangle(credits_bg, (credits_x, credits_y), (self.width - 10, self.height - 10), (45, 42, 38), -1)
        cv2.addWeighted(credits_bg, 0.75, frame, 0.25, 0, frame)
        
        # Add the text
        for i, text in enumerate(credits):
            cv2.putText(
                frame,
                text,
                (credits_x + 10, credits_y + 20 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (230, 225, 220),
                1,
                lineType=cv2.LINE_AA,
            )

    def draw_cursor(
        self,
        frame: np.ndarray,
        point: Tuple[int, int],
        drawing: bool,
        is_eraser: bool = False,
    ) -> None:
        """Visual fingertip cursor — filled when drawing, ring when hovering."""
        color = self.current_color if drawing else (255, 255, 255)
        if drawing and not is_eraser:
            cv2.circle(frame, point, 8, color, -1, lineType=cv2.LINE_AA)
        elif is_eraser:
            cv2.circle(frame, point, self.brush_size * 4, (100, 100, 255), 2, lineType=cv2.LINE_AA)
            cv2.putText(frame, "Eraser", (point[0]+20, point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 255), 1, cv2.LINE_AA)
        else:
            cv2.circle(frame, point, 10, color, 2, lineType=cv2.LINE_AA)
            cv2.circle(frame, point, 3, color, -1, lineType=cv2.LINE_AA)
