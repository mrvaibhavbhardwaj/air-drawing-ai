"""
Hand tracking module using MediaPipe Hands.

Wraps MediaPipe to detect hand landmarks, count raised fingers,
and return the index fingertip position for air-drawing gestures.
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import cv2
import mediapipe as mp


@dataclass
class HandState:
    """Snapshot of one detected hand used by the drawing app."""

    index_tip: Tuple[int, int]          # Pixel coords of index fingertip
    raised_finger_count: int            # Extended fingers (excluding thumb)
    is_drawing_gesture: bool            # Index only raised -> draw
    is_pause_gesture: bool              # Index + middle raised -> stop / select


class HandTracker:
    """Real-time hand landmark detection via MediaPipe Hands."""

    def __init__(
        self,
        max_hands: int = 1,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.6,
    ) -> None:
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    def process(self, frame_bgr: Any) -> Optional[HandState]:
        """
        Run hand detection on a BGR frame.

        Returns HandState for the first detected hand, or None if no hand found.
        """
        # MediaPipe expects RGB input.
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self._hands.process(frame_rgb)
        frame_rgb.flags.writeable = True

        if not results.multi_hand_landmarks:
            return None

        landmarks = results.multi_hand_landmarks[0]
        height, width = frame_bgr.shape[:2]

        index_tip = self._landmark_to_pixel(landmarks.landmark[8], width, height)
        fingers = self._finger_states(landmarks)
        raised_count = sum(fingers.values())

        # Precise gestures: index-only draws, index+middle pauses for toolbar use.
        index_up = fingers["index"]
        middle_up = fingers["middle"]
        others_down = not fingers["ring"] and not fingers["pinky"]

        is_drawing = index_up and not middle_up and others_down
        is_pause = index_up and middle_up and others_down

        return HandState(
            index_tip=index_tip,
            raised_finger_count=raised_count,
            is_drawing_gesture=is_drawing,
            is_pause_gesture=is_pause,
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

        Thumb is ignored because it often triggers accidentally during pointing.
        """
        lm = landmarks.landmark

        def is_up(tip_idx: int, pip_idx: int) -> bool:
            return lm[tip_idx].y < lm[pip_idx].y

        return {
            "index": is_up(8, 6),
            "middle": is_up(12, 10),
            "ring": is_up(16, 14),
            "pinky": is_up(20, 18),
        }

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._hands.close()
