"""
Hand tracking module using MediaPipe Hands (Tasks API).

Wraps MediaPipe to detect hand landmarks, count raised fingers,
and return the index fingertip position for air-drawing gestures.
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple
import os

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

@dataclass
class HandState:
    """Snapshot of one detected hand used by the drawing app."""
    index_tip: Tuple[int, int]          # Pixel coords of index fingertip
    raised_finger_count: int            # Extended fingers (excluding thumb)
    is_drawing_gesture: bool            # Index only raised -> draw
    is_pause_gesture: bool              # Index + middle raised -> stop / select
    is_eraser_gesture: bool             # Index + middle + ring raised -> erase


class HandTracker:
    """Real-time hand landmark detection via MediaPipe Tasks."""

    def __init__(
        self,
        max_hands: int = 1,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.6,
    ) -> None:
        model_path = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def process(self, frame_bgr: Any) -> Optional[HandState]:
        """
        Run hand detection on a BGR frame.

        Returns HandState for the first detected hand, or None if no hand found.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        results = self._landmarker.detect(mp_image)

        if not results.hand_landmarks:
            return None

        # Take the first hand
        landmarks = results.hand_landmarks[0]
        height, width = frame_bgr.shape[:2]

        index_tip = self._landmark_to_pixel(landmarks[8], width, height)
        fingers = self._finger_states(landmarks)
        raised_count = sum(fingers.values())

        index_up = fingers["index"]
        middle_up = fingers["middle"]
        ring_up = fingers["ring"]
        pinky_up = fingers["pinky"]

        is_drawing = index_up and not middle_up and not ring_up and not pinky_up
        is_pause = index_up and middle_up and not ring_up and not pinky_up
        is_eraser = index_up and middle_up and ring_up and not pinky_up

        return HandState(
            index_tip=index_tip,
            raised_finger_count=raised_count,
            is_drawing_gesture=is_drawing,
            is_pause_gesture=is_pause,
            is_eraser_gesture=is_eraser,
        )

    @staticmethod
    def _landmark_to_pixel(landmark: Any, width: int, height: int) -> Tuple[int, int]:
        """Convert normalized [0, 1] landmark coords to pixel coordinates."""
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        return x, y

    def _finger_states(self, landmarks: Any) -> dict:
        """
        Return whether index/middle/ring/pinky are extended.
        Thumb is ignored.
        """
        wrist = landmarks[0]

        def is_extended(tip_idx: int, pip_idx: int, mcp_idx: int) -> bool:
            tip = landmarks[tip_idx]
            pip = landmarks[pip_idx]
            mcp = landmarks[mcp_idx]

            tip_vec = np.array([tip.x - pip.x, tip.y - pip.y])
            pip_vec = np.array([pip.x - mcp.x, pip.y - mcp.y])
            if np.linalg.norm(tip_vec) < 1e-6 or np.linalg.norm(pip_vec) < 1e-6:
                return False

            cosine = np.dot(tip_vec, pip_vec) / (np.linalg.norm(tip_vec) * np.linalg.norm(pip_vec))
            is_straight = cosine > 0.55
            is_farther_from_wrist = self._distance_to_wrist(tip, wrist) > self._distance_to_wrist(pip, wrist)

            return is_straight and is_farther_from_wrist

        return {
            "index": is_extended(8, 6, 5),
            "middle": is_extended(12, 10, 9),
            "ring": is_extended(16, 14, 13),
            "pinky": is_extended(20, 18, 17),
        }

    @staticmethod
    def _distance_to_wrist(landmark: Any, wrist: Any) -> float:
        dx = landmark.x - wrist.x
        dy = landmark.y - wrist.y
        return np.hypot(dx, dy)

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._landmarker.close()
