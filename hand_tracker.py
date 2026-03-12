import cv2
import mediapipe as mp


class HandTracker:
    def __init__(
        self,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.55,
        process_scale=0.75,
    ):
        self.process_scale = process_scale
        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.connections = list(self.mp_hands.HAND_CONNECTIONS)

    def process_frame(self, frame):
        """
        Returns a list of detected hands. Each item contains:
            {
                "label": "Left" or "Right",
                "landmarks": [(x, y), ... 21 items ...],
                "bbox": (x1, y1, x2, y2),
                "center": (cx, cy),
                "area": int,
            }
        """
        h, w = frame.shape[:2]

        if self.process_scale != 1.0:
            small = cv2.resize(
                frame,
                (int(w * self.process_scale), int(h * self.process_scale)),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            small = frame

        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        rgb.flags.writeable = True

        hands_data = []

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                label = handedness.classification[0].label

                lm_list = []
                xs = []
                ys = []

                for lm in hand_landmarks.landmark:
                    px = int(lm.x * w)
                    py = int(lm.y * h)
                    px = max(0, min(w - 1, px))
                    py = max(0, min(h - 1, py))
                    lm_list.append((px, py))
                    xs.append(px)
                    ys.append(py)

                x1, y1 = min(xs), min(ys)
                x2, y2 = max(xs), max(ys)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                area = max(1, (x2 - x1) * (y2 - y1))

                hands_data.append(
                    {
                        "label": label,
                        "landmarks": lm_list,
                        "bbox": (x1, y1, x2, y2),
                        "center": (cx, cy),
                        "area": area,
                    }
                )

        hands_data.sort(key=lambda item: item["area"], reverse=True)
        return hands_data

    def draw_hands(self, frame, hands_data):
        for hand in hands_data:
            lm = hand["landmarks"]

            for start_idx, end_idx in self.connections:
                x1, y1 = lm[start_idx]
                x2, y2 = lm[end_idx]
                cv2.line(frame, (x1, y1), (x2, y2), (110, 255, 110), 2, cv2.LINE_AA)

            for idx, (x, y) in enumerate(lm):
                radius = 4 if idx in (4, 8, 12, 16, 20) else 3
                cv2.circle(frame, (x, y), radius, (255, 80, 255), -1, cv2.LINE_AA)

    def close(self):
        self.hands.close()