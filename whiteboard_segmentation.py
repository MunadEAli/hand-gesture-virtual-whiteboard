import cv2
import mediapipe as mp
import numpy as np


class WhiteboardSegmentation:
    def __init__(
        self,
        model_selection=1,
        threshold=0.58,
        scale=0.45,
        update_every_n_frames=2,
        temporal_alpha=0.55,
        edge_width=0.16,
    ):
        self.threshold = threshold
        self.scale = scale
        self.update_every_n_frames = update_every_n_frames
        self.temporal_alpha = temporal_alpha
        self.edge_width = edge_width

        self.frame_count = 0
        self.last_mask = None

        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.segmenter = self.mp_selfie_segmentation.SelfieSegmentation(
            model_selection=model_selection
        )

    def _compute_mask(self, frame):
        h, w = frame.shape[:2]
        sw = max(1, int(w * self.scale))
        sh = max(1, int(h * self.scale))

        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        rgb_small.flags.writeable = False
        results = self.segmenter.process(rgb_small)
        rgb_small.flags.writeable = True

        if results.segmentation_mask is None:
            return None

        mask_small = results.segmentation_mask.astype(np.float32)
        mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_LINEAR)

        mask_u8 = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
        mask_u8 = cv2.bilateralFilter(mask_u8, 7, 45, 45)

        kernel = np.ones((3, 3), np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = mask_u8.astype(np.float32) / 255.0

        if self.last_mask is not None:
            mask = self.temporal_alpha * mask + (1.0 - self.temporal_alpha) * self.last_mask

        return np.clip(mask, 0.0, 1.0)

    def apply_white_background(self, frame):
        self.frame_count += 1
        should_update = self.last_mask is None or self.frame_count % self.update_every_n_frames == 0

        if should_update:
            mask = self._compute_mask(frame)
            if mask is not None:
                self.last_mask = mask

        if self.last_mask is None:
            return frame

        alpha = np.clip((self.last_mask - self.threshold) / max(1e-6, self.edge_width), 0.0, 1.0)
        alpha = alpha[..., None]

        frame_f = frame.astype(np.float32)
        white = np.full_like(frame_f, 255.0)
        output = frame_f * alpha + white * (1.0 - alpha)
        return np.clip(output, 0, 255).astype(np.uint8)

    def reset_cache(self):
        self.last_mask = None
        self.frame_count = 0

    def close(self):
        self.segmenter.close()