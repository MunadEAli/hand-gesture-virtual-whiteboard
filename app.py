import math
import platform
import time

import cv2
import numpy as np

from drawing_canvas import DrawingCanvas
from gesture_logic import (
    GestureStabilizer,
    ToggleLatch,
    classify_gesture,
    fingers_up,
    format_fingers,
)
from hand_tracker import HandTracker


CAMERA_INDEX = 0
FRAME_WIDTH = 960
FRAME_HEIGHT = 540
WINDOW_NAME = "Hand Whiteboard - CPU Smooth"

# Whiteboard toggle:
# both hands open, hold briefly -> toggle ON/OFF
WHITEBOARD_TOGGLE_HOLD_SECONDS = 0.9
WHITEBOARD_MIN_HAND_SEPARATION_RATIO = 0.25  # hands must be apart horizontally

# Anti-mess draw entry:
# do not start drawing immediately while opening the hand
DRAW_ENTRY_STILL_FRAMES = 3
DRAW_ENTRY_SPEED_THRESHOLD = 12.0  # pixels/frame


def create_camera(index=0):
    if platform.system().lower().startswith("win"):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(index)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except cv2.error:
        pass

    return cap


def draw_toolbar(frame, drawing, whiteboard_mode):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 96), (28, 28, 28), -1)
    frame[:] = cv2.addWeighted(overlay, 0.78, frame, 0.22, 0)

    title = (
        "1 finger: draw | 2: blue | 3: green | 4: highlight | "
        "5: erase | both hands open: whiteboard toggle"
    )
    cv2.putText(
        frame,
        title,
        (18, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )

    tool_text = f"Tool: {drawing.current_tool}"
    mode_text = f"Whiteboard: {'ON' if whiteboard_mode else 'OFF'}"

    cv2.putText(
        frame,
        tool_text,
        (20, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        mode_text,
        (240, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255) if whiteboard_mode else (180, 180, 180),
        2,
        cv2.LINE_AA,
    )


def draw_status_panel(frame, fps, raw_gesture, stable_gesture, finger_text, info_text):
    x1 = frame.shape[1] - 235
    y1 = 110
    x2 = frame.shape[1] - 12
    y2 = 215

    panel = frame.copy()
    cv2.rectangle(panel, (x1, y1), (x2, y2), (18, 18, 18), -1)
    frame[:] = cv2.addWeighted(panel, 0.72, frame, 0.28, 0)

    lines = [
        (f"FPS: {int(fps)}", (80, 255, 80)),
        (f"Raw: {raw_gesture}", (180, 200, 255)),
        (f"Stable: {stable_gesture}", (255, 255, 255)),
        (f"Info: {info_text}", (180, 255, 180)),
    ]

    y = y1 + 22
    for text, color in lines:
        cv2.putText(
            frame,
            text,
            (x1 + 8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            color,
            1,
            cv2.LINE_AA,
        )
        y += 22

    cv2.putText(
        frame,
        "Q / ESC",
        (x1 + 8, y2 - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )


def draw_index_cursor(frame, point, color=(0, 0, 255)):
    if point is None:
        return
    x, y = point
    cv2.circle(frame, (x, y), 10, color, -1, cv2.LINE_AA)
    cv2.circle(frame, (x, y), 18, color, 2, cv2.LINE_AA)


def draw_eraser_preview(frame, bbox, pad=18):
    x1, y1, x2, y2 = bbox
    x1 -= pad
    y1 -= pad
    x2 += pad
    y2 += pad
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 3, cv2.LINE_AA)


def window_closed(window_name):
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def is_both_hands_open_toggle(hands_data):
    """
    Rare gesture for whiteboard toggle:
    - exactly 2 hands
    - both hands show all 5 fingers open
    - hands are reasonably far apart horizontally
    """
    if len(hands_data) != 2:
        return False

    h1, h2 = hands_data[0], hands_data[1]
    f1 = fingers_up(h1["landmarks"], h1["label"])
    f2 = fingers_up(h2["landmarks"], h2["label"])

    both_open = (f1 == [1, 1, 1, 1, 1]) and (f2 == [1, 1, 1, 1, 1])
    if not both_open:
        return False

    x_sep = abs(h1["center"][0] - h2["center"][0])
    min_sep = int(FRAME_WIDTH * WHITEBOARD_MIN_HAND_SEPARATION_RATIO)

    return x_sep >= min_sep


def main():
    cap = create_camera(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    tracker = HandTracker(
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.55,
        process_scale=0.75,
    )
    drawing = DrawingCanvas(FRAME_WIDTH, FRAME_HEIGHT)
    stabilizer = GestureStabilizer(confirm_frames=3, action_warmup_frames=2)

    whiteboard_toggle_start = None
    whiteboard_toggle_latch = ToggleLatch()

    whiteboard_mode = False
    info_text = "RUNNING"

    prev_index_raw = None
    draw_entry_counter = 0
    in_draw_session = False

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, FRAME_WIDTH, FRAME_HEIGHT)

    prev_time = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("ERROR: Failed to read frame.")
                break

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(
                frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_LINEAR
            )

            hands_data = tracker.process_frame(frame)

            raw_gesture = "NO_HAND"
            stable_gesture = stabilizer.stable
            finger_text = "[]"
            preview_point = None

            if whiteboard_mode:
                output_frame = np.full_like(frame, 255)
            else:
                output_frame = frame.copy()

            both_open_toggle_active = is_both_hands_open_toggle(hands_data)

            if both_open_toggle_active:
                if whiteboard_toggle_start is None:
                    whiteboard_toggle_start = time.perf_counter()

                hold_time = time.perf_counter() - whiteboard_toggle_start
                info_text = f"BOTH HANDS OPEN HOLD: {hold_time:.1f}s"

                if (
                    hold_time >= WHITEBOARD_TOGGLE_HOLD_SECONDS
                    and whiteboard_toggle_latch.trigger(True)
                ):
                    whiteboard_mode = not whiteboard_mode
                    drawing.clear()
                    drawing.reset_pointer()

                    in_draw_session = False
                    draw_entry_counter = 0
                    prev_index_raw = None

                    info_text = (
                        "WHITEBOARD ON + CLEARED"
                        if whiteboard_mode
                        else "WHITEBOARD OFF + CLEARED"
                    )
            else:
                whiteboard_toggle_start = None
                whiteboard_toggle_latch.trigger(False)

            primary_hand = hands_data[0] if hands_data else None

            if primary_hand is not None:
                primary_fingers = fingers_up(
                    primary_hand["landmarks"], primary_hand["label"]
                )
                raw_gesture = classify_gesture(primary_fingers)
                stable_gesture = stabilizer.update(raw_gesture)
                finger_text = format_fingers(primary_fingers)

                lm = primary_hand["landmarks"]
                ix, iy = lm[8]

                if prev_index_raw is None:
                    raw_speed = 999.0
                else:
                    raw_speed = math.hypot(ix - prev_index_raw[0], iy - prev_index_raw[1])
                prev_index_raw = (ix, iy)

                # Prevent both-hands-open whiteboard gesture from becoming eraser
                if not both_open_toggle_active:
                    if stabilizer.changed and stable_gesture in {
                        "SELECT_BLUE",
                        "SELECT_GREEN",
                        "SELECT_HIGHLIGHTER",
                        "SELECT_ERASER",
                    }:
                        drawing.set_tool_from_gesture(stable_gesture)
                        info_text = f"TOOL -> {drawing.current_tool}"

                    ts = time.perf_counter()

                    # ERASER
                    if (
                        drawing.current_tool == "ERASER"
                        and stabilizer.eraser_ready(raw_gesture)
                    ):
                        in_draw_session = False
                        draw_entry_counter = 0
                        drawing.erase_with_hand_bbox(primary_hand["bbox"])
                        preview_point = primary_hand["center"]
                        info_text = "ERASING"

                    # DRAW / HIGHLIGHTER
                    elif drawing.current_tool in {"PEN", "HIGHLIGHTER"}:
                        if stable_gesture == "ACTION" and raw_gesture == "ACTION":
                            if not in_draw_session:
                                # only ENTER drawing after the hand has settled briefly
                                if raw_speed <= DRAW_ENTRY_SPEED_THRESHOLD:
                                    draw_entry_counter += 1
                                else:
                                    draw_entry_counter = 0

                                if (
                                    stabilizer.action_ready(raw_gesture)
                                    and draw_entry_counter >= DRAW_ENTRY_STILL_FRAMES
                                ):
                                    in_draw_session = True
                                    drawing.reset_pointer()

                            if in_draw_session:
                                preview_point = drawing.draw_action(ix, iy, ts)
                                info_text = f"DRAWING: {drawing.current_tool}"
                            else:
                                drawing.reset_pointer()
                                preview_point = (ix, iy)
                                info_text = "READY TO DRAW"
                        else:
                            # Stop immediately on any exit from ACTION
                            in_draw_session = False
                            draw_entry_counter = 0
                            drawing.reset_pointer()

                            if stable_gesture == "IDLE":
                                info_text = "IDLE"
                            elif stable_gesture == "THUMB_ONLY":
                                info_text = "THUMB DETECTED"
                            elif stable_gesture == "UNKNOWN":
                                info_text = "UNKNOWN GESTURE"
                            else:
                                info_text = "NOT DRAWING"

                    else:
                        in_draw_session = False
                        draw_entry_counter = 0
                        drawing.reset_pointer()
                        info_text = f"TOOL READY: {drawing.current_tool}"

                else:
                    in_draw_session = False
                    draw_entry_counter = 0
                    drawing.reset_pointer()

            else:
                stabilizer.update("NO_HAND")
                drawing.reset_pointer()
                prev_index_raw = None
                in_draw_session = False
                draw_entry_counter = 0
                info_text = "NO HAND"

            output_frame = drawing.overlay_on_frame(output_frame)
            tracker.draw_hands(output_frame, hands_data)

            if primary_hand is not None:
                if both_open_toggle_active:
                    for hand in hands_data:
                        draw_eraser_preview(output_frame, hand["bbox"], pad=10)
                elif (
                    drawing.current_tool == "ERASER"
                    and stabilizer.eraser_ready(raw_gesture)
                ):
                    draw_eraser_preview(output_frame, primary_hand["bbox"])
                elif preview_point is not None:
                    draw_index_cursor(output_frame, preview_point)
                else:
                    draw_index_cursor(output_frame, primary_hand["landmarks"][8])

            current_time = time.perf_counter()
            fps = 1.0 / max(1e-6, (current_time - prev_time))
            prev_time = current_time

            draw_toolbar(output_frame, drawing, whiteboard_mode)
            draw_status_panel(
                output_frame,
                fps,
                raw_gesture,
                stable_gesture,
                finger_text,
                info_text,
            )

            cv2.imshow(WINDOW_NAME, output_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if window_closed(WINDOW_NAME):
                break

    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()