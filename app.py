import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import queue
import sounddevice as sd
import vosk
import json
import time
import threading
import pyautogui
import platform
import subprocess
import numpy as np
import easyocr

# ============================================================
# 1. PLATFORM & ACTIVE APP DETECTION
# ============================================================
IS_MAC = platform.system() == "Darwin"

if IS_MAC:
    def get_active_app():
        script = 'tell application "System Events" to get name of first application process whose frontmost is true'
        result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
        return result.stdout.strip()
else:
    import pygetwindow as gw
    def get_active_app():
        active = gw.getActiveWindow()
        return active.title if active else "Unknown"

# ============================================================
# 2. CONFIG & SAFE EXECUTOR
# ============================================================
with open("config.json") as f:
    config = json.load(f)

def execute_action(action_name):
    active_app = get_active_app()
    app_config = config.get(active_app, config.get("default", {}))

    if action_name not in app_config:
        app_config = config.get("default", {})
    if action_name not in app_config:
        print(f"Unknown action: {action_name}")
        return

    key_combo = app_config[action_name]

    if key_combo == "unsupported":
        print(f"'{action_name}' is not supported in {active_app}. Skipping.")
        return

    if key_combo == "scroll_down":
        w, h = pyautogui.size()
        pyautogui.moveTo(w // 2, h // 2)
        pyautogui.scroll(-600)
    elif key_combo == "scroll_up":
        w, h = pyautogui.size()
        pyautogui.moveTo(w // 2, h // 2)
        pyautogui.scroll(600)
    elif "+" in key_combo:
        keys = key_combo.split("+")
        if IS_MAC:
            keys = ["command" if k == "ctrl" else k for k in keys]
        pyautogui.hotkey(*keys)
    else:
        pyautogui.press(key_combo)

    print(f"Executed: {action_name} in {active_app} (trigger: {key_combo})")

COOLDOWN = 1.0
last_trigger_time = 0
trigger_lock = threading.Lock()

def safe_execute(action_name, source):
    global last_trigger_time
    with trigger_lock:
        now = time.time()
        if now - last_trigger_time < COOLDOWN:
            return
        last_trigger_time = now
    print(f"[{source}] -> {action_name}")
    execute_action(action_name)

# ============================================================
# 3. ON-SCREEN BUTTON TARGET ACQUISITION (OCR & OVERLAY)
# ============================================================
print("Initializing on-device EasyOCR engine...")
ocr_reader = easyocr.Reader(['en'], gpu=False)

overlay_active = False
overlay_targets = {}  # {number: (click_x, click_y, "button text")}
overlay_lock = threading.Lock()

def trigger_screen_scan():
    global overlay_active, overlay_targets
    with overlay_lock:
        if overlay_active:
            return
        print("\n[Scanning Screen for Clickable Buttons...]")

        screen_img = pyautogui.screenshot()
        img_np = np.array(screen_img)
        frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        img_h, img_w, _ = frame.shape
        screen_w, screen_h = pyautogui.size()

        # Handle Retina & Windows DPI coordinate scaling
        scale_x = screen_w / img_w
        scale_y = screen_h / img_h

        results = ocr_reader.readtext(frame)
        valid = []
        for bbox, text, conf in results:
            cleaned = text.strip().lower()
            if conf > 0.35 and len(cleaned) > 1:
                valid.append((bbox, cleaned, conf))

        valid = valid[:9]  # Keep the top 9 detected targets

        overlay_targets = {}
        for idx, (bbox, text, conf) in enumerate(valid, start=1):
            (tl, tr, br, bl) = bbox
            cx = int((tl[0] + br[0]) / 2)
            cy = int((tl[1] + br[1]) / 2)

            click_x = int(cx * scale_x)
            click_y = int(cy * scale_y)
            overlay_targets[idx] = (click_x, click_y, text)

            # Draw visual highlight boxes and red numbered badges
            pt1 = (int(tl[0]), int(tl[1]))
            pt2 = (int(br[0]), int(br[1]))
            cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)

            badge_pos = (int(tl[0]) - 5, int(tl[1]) - 5)
            cv2.circle(frame, badge_pos, 18, (0, 0, 255), -1)
            cv2.putText(frame, str(idx), (badge_pos[0] - 7, badge_pos[1] + 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if overlay_targets:
            overlay_active = True
            preview = cv2.resize(frame, (screen_w // 2, screen_h // 2))
            cv2.imshow("OmiControl Targets", preview)
            print("Detected Targets on Screen:")
            for num, (x, y, text) in overlay_targets.items():
                print(f"  [{num}] '{text}' at Screen Pos ({x}, {y})")
        else:
            print("No button text found.")

def click_target(target_index):
    global overlay_active
    with overlay_lock:
        if not overlay_active:
            return
        if target_index in overlay_targets:
            x, y, text = overlay_targets[target_index]
            print(f"\nClicking [{target_index}] '{text}' at ({x}, {y})")
            pyautogui.click(x, y)
        overlay_active = False
        try:
            cv2.destroyWindow("OmiControl Targets")
        except Exception:
            pass

def match_target_by_name(text_input):
    with overlay_lock:
        if not overlay_active:
            return None
        text_input = text_input.lower().strip()
        for idx, (_, _, target_text) in overlay_targets.items():
            if target_text in text_input or text_input in target_text:
                return idx
    return None

# ============================================================
# 4. VOICE ENGINE (ASR + PREFIX-BASED CLICK DISPATCH)
# ============================================================
WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9
}

def voice_to_action(text):
    text = text.lower()
    if "scan" in text:
        return "scan_screen"
    elif "next" in text or ("scroll" in text and "down" in text):
        return "next_slide"
    elif "back" in text or ("scroll" in text and "up" in text):
        return "previous_slide"
    elif "mute" in text:
        return "mute"
    elif "underline" in text:
        return "underline"
    return None

def run_voice_loop():
    model = vosk.Model("vosk-model-small-en-us-0.15")
    samplerate = 16000
    q = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        q.put(bytes(indata))

    # Extended grammar to distinguish standard commands from button-clicks
    vocabulary = [
        "omi", "click", "scan", "next", "back", "mute", "start", "stop",
        "underline", "scroll", "up", "down",
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"
    ]
    recognizer = vosk.KaldiRecognizer(model, samplerate, json.dumps(vocabulary))

    WAKE_WORD = "omi"
    ACTION_WINDOW = 5.0
    awaiting_command = False
    window_start_time = 0
    command_buffer = ""

    print("Voice ready: say 'omi' followed by a command.")

    with sd.RawInputStream(samplerate=samplerate, blocksize=4000, dtype='int16',
                           channels=1, callback=audio_callback):
        while True:
            data = q.get()

            if awaiting_command and (time.time() - window_start_time > ACTION_WINDOW):
                awaiting_command = False
                command_buffer = ""
                print("Voice window expired.")

            is_final = recognizer.AcceptWaveform(data)

            if not awaiting_command:
                partial_text = json.loads(recognizer.PartialResult()).get("partial", "")
                if WAKE_WORD in partial_text:
                    awaiting_command = True
                    window_start_time = time.time()
                    command_buffer = ""
                    recognizer.Reset()
                    print("Wake word detected.")
            else:
                if is_final:
                    final_text = json.loads(recognizer.Result()).get("text", "")
                    if final_text:
                        command_buffer = (command_buffer + " " + final_text).strip()
                    current = command_buffer
                else:
                    partial_text = json.loads(recognizer.PartialResult()).get("partial", "")
                    current = (command_buffer + " " + partial_text).strip()

                if current:
                    # Case 1: Button Click Command ("omi click ...")
                    if "click" in current:
                        target_phrase = current.split("click", 1)[1].strip()
                        clicked = False

                        # Number badge match (e.g., "omi click three")
                        for word, num in WORD_TO_NUM.items():
                            if word in target_phrase:
                                click_target(num)
                                clicked = True
                                break

                        # Button text name match (e.g., "omi click next")
                        if not clicked and target_phrase:
                            matched_num = match_target_by_name(target_phrase)
                            if matched_num:
                                click_target(matched_num)
                                clicked = True
                            else:
                                print(f"No button found matching: '{target_phrase}'")

                        awaiting_command = False
                        command_buffer = ""
                        continue

                    # Case 2: Standard Slide/System Command ("omi next", "omi back")
                    action = voice_to_action(current)
                    if action == "scan_screen":
                        trigger_screen_scan()
                        awaiting_command = False
                        command_buffer = ""
                    elif action:
                        safe_execute(action, "voice")
                        awaiting_command = False
                        command_buffer = ""

# ============================================================
# 5. VISION ENGINE (GESTURES & NUMBERED SELECTION)
# ============================================================
def count_fingers(hand_landmarks):
    # Tip IDs: 8 (index), 12 (middle), 16 (ring), 20 (pinky)
    tips = [8, 12, 16, 20]
    count = sum(1 for tip in tips if hand_landmarks[tip].y < hand_landmarks[tip - 2].y)

    # Thumb extension relative to MCP knuckle
    thumb_is_open = hand_landmarks[4].x < hand_landmarks[3].x if hand_landmarks[17].x > hand_landmarks[5].x else hand_landmarks[4].x > hand_landmarks[3].x
    if thumb_is_open:
        count += 1
    return count

def run_vision_loop():
    base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7
    )
    detector = vision.HandLandmarker.create_from_options(options)

    camera_index = 0
    MAX_CAMERAS = 4
    cap = cv2.VideoCapture(camera_index)

    STABILITY_FRAMES = 5
    pending_fingers = None
    pending_count = 0

    print("Vision ready. Press 'q' to quit, 'c' to switch camera.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        detected_fingers = None

        if result.hand_landmarks:
            h, w, _ = frame.shape
            for hand_landmarks in result.hand_landmarks:
                for lm in hand_landmarks:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1)
                detected_fingers = count_fingers(hand_landmarks)
                cv2.putText(frame, f"Fingers: {detected_fingers}", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        if detected_fingers == pending_fingers:
            pending_count += 1
        else:
            pending_fingers = detected_fingers
            pending_count = 1

        if detected_fingers is not None and pending_count >= STABILITY_FRAMES:
            # When overlay is active, finger count selects numbered target badge
            if overlay_active and 1 <= detected_fingers <= 9:
                click_target(detected_fingers)
                pending_count = 0
            else:
                # Standard Presentation Gestures
                if detected_fingers == 5:
                    trigger_screen_scan()
                    pending_count = 0
                elif detected_fingers == 1:
                    safe_execute("next_slide", "gesture")
                elif detected_fingers == 2:
                    safe_execute("previous_slide", "gesture")
                elif detected_fingers == 0:
                    safe_execute("mute", "gesture")

        cv2.imshow("OmiControl Vision", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c'):
            cap.release()
            camera_index = (camera_index + 1) % MAX_CAMERAS
            cap = cv2.VideoCapture(camera_index)
            print(f"Switched to camera index {camera_index}")

    cap.release()
    cv2.destroyAllWindows()

# ============================================================
# 6. MASTER THREAD DISPATCH
# ============================================================
voice_thread = threading.Thread(target=run_voice_loop, daemon=True)
voice_thread.start()

run_vision_loop()