FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]


class GestureStabilizer:
    """
    Low-latency gesture stabilizer.

    - A gesture becomes stable after it is seen for `confirm_frames` consecutive frames.
    - Drawing is blocked for the first few frames after switching into ACTION so half-open
      hands do not accidentally draw.
    - Drawing is also blocked immediately when the raw gesture is no longer ACTION.
    """

    def __init__(self, confirm_frames=3, action_warmup_frames=2):
        self.confirm_frames = confirm_frames
        self.action_warmup_frames = action_warmup_frames

        self.candidate = None
        self.candidate_count = 0

        self.stable = "NO_HAND"
        self.stable_age = 0
        self.changed = False

    def update(self, raw_gesture):
        self.changed = False

        if raw_gesture == self.candidate:
            self.candidate_count += 1
        else:
            self.candidate = raw_gesture
            self.candidate_count = 1

        if self.candidate != self.stable and self.candidate_count >= self.confirm_frames:
            self.stable = self.candidate
            self.stable_age = 0
            self.changed = True
        elif raw_gesture == self.stable:
            self.stable_age += 1

        return self.stable

    def action_ready(self, raw_gesture):
        return (
            self.stable == "ACTION"
            and raw_gesture == "ACTION"
            and self.stable_age >= self.action_warmup_frames
        )

    def eraser_ready(self, raw_gesture):
        return self.stable == "SELECT_ERASER" and raw_gesture == "SELECT_ERASER"


class ToggleLatch:
    """
    Fires once per gesture hold until the gesture is released.
    Useful for thumb-based mode toggles.
    """

    def __init__(self):
        self.armed = True

    def trigger(self, active):
        if active and self.armed:
            self.armed = False
            return True
        if not active:
            self.armed = True
        return False


THUMB_TIP = 4
THUMB_IP = 3
INDEX_TIP = 8
INDEX_PIP = 6
MIDDLE_TIP = 12
MIDDLE_PIP = 10
RING_TIP = 16
RING_PIP = 14
PINKY_TIP = 20
PINKY_PIP = 18


def fingers_up(landmarks, hand_label):
    """
    Returns finger state as [thumb, index, middle, ring, pinky].

    The webcam frame is mirrored in app.py, so thumb logic uses handedness-aware x checks.
    Other fingers use the standard tip-vs-pip y comparison.
    """
    fingers = [0, 0, 0, 0, 0]

    if landmarks[INDEX_TIP][1] < landmarks[INDEX_PIP][1]:
        fingers[1] = 1
    if landmarks[MIDDLE_TIP][1] < landmarks[MIDDLE_PIP][1]:
        fingers[2] = 1
    if landmarks[RING_TIP][1] < landmarks[RING_PIP][1]:
        fingers[3] = 1
    if landmarks[PINKY_TIP][1] < landmarks[PINKY_PIP][1]:
        fingers[4] = 1

    thumb_tip_x = landmarks[THUMB_TIP][0]
    thumb_ip_x = landmarks[THUMB_IP][0]

    if hand_label == "Right":
        if thumb_tip_x < thumb_ip_x:
            fingers[0] = 1
    else:
        if thumb_tip_x > thumb_ip_x:
            fingers[0] = 1

    return fingers


def classify_gesture(fingers):
    """
    Gesture map:
    [thumb, index, middle, ring, pinky]
    """
    if fingers == [0, 0, 0, 0, 0]:
        return "IDLE"
    if fingers == [0, 1, 0, 0, 0]:
        return "ACTION"
    if fingers == [0, 1, 1, 0, 0]:
        return "SELECT_BLUE"
    if fingers == [0, 1, 1, 1, 0]:
        return "SELECT_GREEN"
    if fingers == [0, 1, 1, 1, 1]:
        return "SELECT_HIGHLIGHTER"
    if fingers == [1, 1, 1, 1, 1]:
        return "SELECT_ERASER"
    if fingers == [1, 0, 0, 0, 0]:
        return "THUMB_ONLY"
    return "UNKNOWN"


def format_fingers(fingers):
    if not fingers:
        return "[]"
    return "[" + ", ".join(
        f"{name}:{state}" for name, state in zip(FINGER_NAMES, fingers)
    ) + "]"