"""
Air Drawing — virtual whiteboard with hand gestures + mouse toolbar control.

Hand gestures:
  - 1 finger (index only)  -> draw on canvas
  - 2 fingers (index+middle) -> pause drawing; point at toolbar to pick colors / clear

Mouse (toolbar only):
  - Click a color swatch -> change drawing color
  - Click Clear -> erase canvas

Keys: q / ESC -> quit
"""

import time
from typing import List, Tuple

import cv2
import numpy as np

from drawing_utils import TOOLBAR_HEIGHT, DrawingCanvas
from hand_tracker import HandTracker

WINDOW_NAME = "Air Draw"

PREFERRED_RESOLUTIONS: List[Tuple[int, int]] = [
    (1920, 1080),
    (1280, 720),
    (1280, 800),
    (960, 540),
]


class FPSCounter:
    def __init__(self, smooth_factor: float = 0.9) -> None:
        self._smooth = smooth_factor
        self._fps = 0.0
        self._prev_time = time.perf_counter()

    def tick(self) -> float:
        now = time.perf_counter()
        dt = now - self._prev_time
        self._prev_time = now
        if dt > 0:
            instant = 1.0 / dt
            self._fps = self._smooth * self._fps + (1.0 - self._smooth) * instant
        return self._fps


class HighQualityCamera:
    def __init__(self) -> None:
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam. Check camera permissions.")

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.width, self.height = self._negotiate_resolution(cap)
        self._cap = cap
        self._buffer = np.empty((self.height, self.width, 3), dtype=np.uint8)
        print(f"Camera: {self.width}x{self.height}")

    @staticmethod
    def _negotiate_resolution(cap: cv2.VideoCapture) -> Tuple[int, int]:
        best_w, best_h = 0, 0
        for target_w, target_h in PREFERRED_RESOLUTIONS:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
            cap.set(cv2.CAP_PROP_FPS, 30)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            actual_h, actual_w = frame.shape[:2]
            if actual_w * actual_h > best_w * best_h:
                best_w, best_h = actual_w, actual_h
            if actual_w >= 1280 and actual_h >= 720:
                break
        if best_w == 0:
            ok, frame = cap.read()
            if ok and frame is not None:
                best_h, best_w = frame.shape[:2]
            else:
                best_w, best_h = 1280, 720
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, best_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, best_h)
        return best_w, best_h

    def read(self) -> Tuple[bool, np.ndarray]:
        ok, raw = self._cap.read()
        if not ok or raw is None:
            return False, self._buffer
        h, w = raw.shape[:2]
        if w != self.width or h != self.height:
            cv2.resize(
                raw,
                (self.width, self.height),
                dst=self._buffer,
                interpolation=cv2.INTER_AREA if w > self.width else cv2.INTER_CUBIC,
            )
        else:
            np.copyto(self._buffer, raw)
        cv2.flip(self._buffer, 1, dst=self._buffer)
        return True, self._buffer

    def release(self) -> None:
        self._cap.release()


def apply_canvas_tint(frame: np.ndarray) -> None:
    tinted = frame.copy()
    cv2.rectangle(
        tinted,
        (0, TOOLBAR_HEIGHT),
        (frame.shape[1], frame.shape[0]),
        (30, 28, 26),
        -1,
    )
    cv2.addWeighted(tinted, 0.10, frame, 0.90, 0, frame)


def on_mouse(event: int, x: int, y: int, flags: int, canvas: DrawingCanvas) -> None:
    """Mouse handler: toolbar clicks for colors and clear; hover highlights."""
    if y <= TOOLBAR_HEIGHT:
        if event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN):
            canvas.set_toolbar_hover((x, y))
        if event == cv2.EVENT_LBUTTONDOWN:
            canvas.end_stroke()
            canvas.handle_toolbar_point((x, y))
    elif event == cv2.EVENT_MOUSEMOVE:
        canvas.set_toolbar_hover(None)


def main() -> None:
    camera = HighQualityCamera()
    # High confidence thresholds restrict bad landmarks which causes jitter
    tracker = HandTracker(detection_confidence=0.8, tracking_confidence=0.85)
    canvas = DrawingCanvas(width=camera.width, height=camera.height)
    fps_counter = FPSCounter()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse, canvas)

    was_drawing = False

    print("Air Draw started.")
    print("  Hand: 1 finger = draw | 2 fingers = pause + point at toolbar")
    print("  Hand: 3 fingers = eraser")
    print("  Mouse: click toolbar colors or Clear anytime")
    print("  Keys: s = save drawing | +/- = brush size | q/ESC = quit")

    while True:
        ok, frame = camera.read()
        if not ok:
            print("Failed to read from webcam.")
            break

        apply_canvas_tint(frame)

        hand = tracker.process(frame)
        mode_label = "Mode: Ready"
        is_drawing = False

        if hand is not None:
            tip = hand.index_tip

            if hand.is_drawing_gesture:
                mode_label = "Mode: Drawing"
                is_drawing = True
                canvas.set_toolbar_hover(None)
                canvas.add_stroke_point(tip)
                
            elif hand.is_eraser_gesture:
                mode_label = "Mode: Eraser"
                is_drawing = True
                canvas.set_toolbar_hover(None)
                canvas.add_stroke_point(tip, is_eraser=True)

            elif hand.is_pause_gesture:
                mode_label = "Mode: Select"
                canvas.end_stroke()
                canvas.set_toolbar_hover(tip)
                canvas.handle_toolbar_point(tip)

            else:
                mode_label = "Mode: Idle"
                canvas.end_stroke()
                canvas.set_toolbar_hover(None)

            canvas.draw_cursor(frame, tip, drawing=is_drawing, is_eraser=hand.is_eraser_gesture)

            if was_drawing and not is_drawing:
                canvas.end_stroke()

            was_drawing = is_drawing
        else:
            canvas.end_stroke()
            was_drawing = False
            mode_label = "Mode: No hand"

        canvas.composite(frame)
        canvas.draw_toolbar(frame)

        fps = fps_counter.tick()
        canvas.draw_status(
            frame,
            fps=fps,
            mode_label=mode_label,
            hand_detected=hand is not None,
            resolution=f"{camera.width}x{camera.height}",
        )

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("s"):
            cv2.imwrite("drawing.png", frame)
            print("Canvas saved as drawing.png")
            # Give some visual feedback
            cv2.putText(frame, "SAVED!", (camera.width//2 - 60, camera.height//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            cv2.imshow(WINDOW_NAME, frame)
            cv2.waitKey(500)
        elif key == ord("+") or key == ord("="):
            canvas.brush_size = min(40, canvas.brush_size + 2)
            print(f"Brush size: {canvas.brush_size}")
        elif key == ord("-") or key == ord("_"):
            canvas.brush_size = max(2, canvas.brush_size - 2)
            print(f"Brush size: {canvas.brush_size}")
        
        # Break if the window is closed by the user
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    camera.release()
    tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
