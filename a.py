"""
Blink to Morse Communicator — Streamlit Interface
---------------------------------------------------
Turns eye blinks (captured via webcam) into Morse code, then decodes
that Morse code into readable text, live, in the browser.

Run with:
    streamlit run app.py
"""

import time
import threading

import av
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# ----------------------------------------------------------------------
# 1. PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Blink to Morse Communicator",
    page_icon="👁️",
    layout="wide",
)

# ----------------------------------------------------------------------
# 2. EYE LANDMARKS & MORSE DICTIONARY (same as the original script)
# ----------------------------------------------------------------------
RIGHT_EYE = [33, 159, 158, 133, 153, 145]
LEFT_EYE = [362, 386, 385, 263, 380, 374]

MORSE_CODE = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.', 'G': '--.',
    'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---',
    'P': '.--.', 'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-', 'U': '..-', 'V': '...-',
    'W': '.--', 'X': '-..-', 'Y': '-.--', 'Z': '--..',
    '0': '-----', '1': '.----', '2': '..---', '3': '...--', '4': '....-',
    '5': '.....', '6': '-....', '7': '--...', '8': '---..', '9': '----.',
}
MORSE_TO_TEXT = {value: key for key, value in MORSE_CODE.items()}


def calculate_ear(eye_points: np.ndarray) -> float:
    """Eye Aspect Ratio — same formula as the original script."""
    vertical_1 = np.linalg.norm(eye_points[1] - eye_points[5])
    vertical_2 = np.linalg.norm(eye_points[2] - eye_points[4])
    horizontal = np.linalg.norm(eye_points[0] - eye_points[3])
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


# ----------------------------------------------------------------------
# 3. VIDEO PROCESSOR
#    All blink -> Morse -> text logic lives here. It runs on a
#    background thread managed by streamlit-webrtc, so state is
#    protected with a lock and read by the main thread for display.
# ----------------------------------------------------------------------
class BlinkMorseProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_draw = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.lock = threading.Lock()

        # Tunable settings (updated live from the sidebar sliders)
        self.ear_threshold = 0.20
        self.long_blink_threshold = 0.60
        self.letter_pause = 1.20
        self.word_pause = 2.50
        self.show_mesh = True

        # Blink / decode state
        self.eyes_closed = False
        self.blink_start_time = None
        self.last_blink_time = None
        self.last_letter_time = None
        self.current_morse = ""
        self.decoded_message = ""

        # Latest readings, for the UI to poll
        self.ear_value = 0.0
        self.eye_status = "NO FACE"

        self._clear_requested = False

    # -- public helpers called from the main Streamlit thread ----------
    def request_clear(self) -> None:
        with self.lock:
            self._clear_requested = True

    def update_settings(self, ear_threshold, long_blink_threshold, letter_pause, word_pause, show_mesh) -> None:
        with self.lock:
            self.ear_threshold = ear_threshold
            self.long_blink_threshold = long_blink_threshold
            self.letter_pause = letter_pause
            self.word_pause = word_pause
            self.show_mesh = show_mesh

    def get_state(self) -> dict:
        with self.lock:
            return {
                "ear": self.ear_value,
                "status": self.eye_status,
                "morse": self.current_morse,
                "message": self.decoded_message,
            }

    def _clear(self) -> None:
        self.decoded_message = ""
        self.current_morse = ""
        self.last_blink_time = None
        self.last_letter_time = None
        self.blink_start_time = None
        self.eyes_closed = False

    # -- main video callback -------------------------------------------
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)

        with self.lock:
            if self._clear_requested:
                self._clear()
                self._clear_requested = False

            eye_status = "NO FACE"

            if result.multi_face_landmarks:
                for face_landmarks in result.multi_face_landmarks:
                    if self.show_mesh:
                        self.mp_draw.draw_landmarks(
                            img,
                            face_landmarks,
                            self.mp_face_mesh.FACEMESH_TESSELATION,
                            self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1),
                            self.mp_draw.DrawingSpec(color=(0, 0, 0), thickness=1),
                        )

                    right_eye_points = []
                    left_eye_points = []

                    for index in RIGHT_EYE:
                        landmark = face_landmarks.landmark[index]
                        px, py = int(landmark.x * w), int(landmark.y * h)
                        right_eye_points.append(np.array([px, py]))
                        cv2.circle(img, (px, py), 3, (0, 255, 0), -1)

                    for index in LEFT_EYE:
                        landmark = face_landmarks.landmark[index]
                        px, py = int(landmark.x * w), int(landmark.y * h)
                        left_eye_points.append(np.array([px, py]))
                        cv2.circle(img, (px, py), 3, (0, 255, 0), -1)

                    right_eye_points = np.array(right_eye_points)
                    left_eye_points = np.array(left_eye_points)

                    right_ear = calculate_ear(right_eye_points)
                    left_ear = calculate_ear(left_eye_points)
                    ear = (right_ear + left_ear) / 2
                    self.ear_value = ear

                    # -- blink detection --
                    if ear < self.ear_threshold:
                        eye_status = "CLOSED"
                        if not self.eyes_closed:
                            self.eyes_closed = True
                            self.blink_start_time = time.time()
                    else:
                        eye_status = "OPEN"
                        if self.eyes_closed:
                            self.eyes_closed = False
                            blink_duration = time.time() - self.blink_start_time
                            if blink_duration < self.long_blink_threshold:
                                self.current_morse += "."
                            else:
                                self.current_morse += "-"
                            self.last_blink_time = time.time()
                            self.blink_start_time = None

                    # -- letter pause: decode current_morse --
                    if not self.eyes_closed and self.last_blink_time is not None:
                        pause = time.time() - self.last_blink_time
                        if pause >= self.letter_pause:
                            if self.current_morse != "":
                                letter = MORSE_TO_TEXT.get(self.current_morse)
                                if letter:
                                    self.decoded_message += letter
                                    self.last_letter_time = time.time()
                                self.current_morse = ""
                            self.last_blink_time = None

                    # -- word pause: append a space --
                    if (
                        self.last_letter_time is not None
                        and self.last_blink_time is None
                        and not self.eyes_closed
                    ):
                        word_pause = time.time() - self.last_letter_time
                        if word_pause >= self.word_pause:
                            if self.decoded_message != "" and not self.decoded_message.endswith(" "):
                                self.decoded_message += " "
                            self.last_letter_time = None

            self.eye_status = eye_status

            # -- on-frame overlay --
            color = (0, 255, 0) if self.eye_status == "OPEN" else (0, 0, 255)
            cv2.putText(img, f"EAR: {self.ear_value:.3f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(img, f"Eyes: {self.eye_status}", (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(img, f"Code: {self.current_morse}", (20, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Decoded message, bottom-left, wrapped so long messages don't run off-frame.
            message_text = self.decoded_message if self.decoded_message else "(no message yet)"
            wrapped_lines = self._wrap_text(f"Message: {message_text}", max_chars=38)
            line_height = 28
            start_y = h - 15 - line_height * (len(wrapped_lines) - 1)
            for i, line in enumerate(wrapped_lines):
                cv2.putText(
                    img, line, (20, start_y + i * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2,
                )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    @staticmethod
    def _wrap_text(text: str, max_chars: int) -> list:
        """Simple word-wrap so the overlay text stays within the frame width."""
        words = text.split(" ")
        lines, current = [], ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if len(candidate) > max_chars and current:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
        return lines or [text]


# ----------------------------------------------------------------------
# 4. SIDEBAR — settings & reference
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    ear_threshold = st.slider(
        "EAR threshold (closed-eye sensitivity)",
        min_value=0.10, max_value=0.35, value=0.20, step=0.01,
        help="Lower = eyes must close more to register as CLOSED.",
    )
    long_blink_threshold = st.slider(
        "Long-blink cutoff — dot vs dash (seconds)",
        min_value=0.30, max_value=1.20, value=0.60, step=0.05,
    )
    letter_pause = st.slider(
        "Letter pause (seconds)",
        min_value=0.5, max_value=2.5, value=1.20, step=0.1,
        help="How long to keep eyes open before the current Morse code is decoded into a letter.",
    )
    word_pause = st.slider(
        "Word pause (seconds)",
        min_value=1.0, max_value=4.0, value=2.50, step=0.1,
        help="How long to keep eyes open after a letter before a space is inserted.",
    )
    show_mesh = st.checkbox("Show face mesh overlay", value=True)

    st.divider()
    st.subheader("📖 Morse Reference")
    ref_cols = st.columns(2)
    items = list(MORSE_CODE.items())
    half = len(items) // 2 + 1
    with ref_cols[0]:
        for k, v in items[:half]:
            st.text(f"{k}   {v}")
    with ref_cols[1]:
        for k, v in items[half:]:
            st.text(f"{k}   {v}")

# ----------------------------------------------------------------------
# 5. MAIN LAYOUT
# ----------------------------------------------------------------------
st.title("👁️ Blink to Morse Communicator")
st.caption("Short blink = dot • Long blink = dash • Pause to complete a letter or word")

col_video, col_status = st.columns([2, 1])

with col_video:
    ctx = webrtc_streamer(
        key="blink-morse",
        video_processor_factory=BlinkMorseProcessor,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": True, "audio": False},
    )

with col_status:
    st.subheader("Live Status")

    # A st.fragment reruns only this function on its own schedule and
    # clears its own element tree before each run. A plain `while` loop
    # calling widgets repeatedly within a single script execution is what
    # causes StreamlitDuplicateElementId — once two iterations produce
    # identical widget arguments (e.g. the Morse code hasn't changed
    # between polls), Streamlit sees two elements with the same
    # auto-generated ID. Fragments avoid that.
    #
    # Note: text_input/text_area are stateful widgets — once given a `key`,
    # Streamlit binds their displayed value to st.session_state on first
    # render and ignores the `value=` argument on every later rerun. That
    # froze the Morse code / message readouts. st.code() has no such
    # binding, so it always shows the latest value passed to it.
    @st.fragment(run_every=0.2)
    def render_status() -> None:
        if ctx.video_processor:
            state = ctx.video_processor.get_state()
            st.metric("EAR", f"{state['ear']:.3f}")
            st.metric("Eye Status", state["status"])
            st.caption("Current Morse Code")
            st.code(state["morse"] if state["morse"] else " ", language=None)
            st.caption("Decoded Message")
            st.code(state["message"] if state["message"] else " ", language=None)
        else:
            st.info("Click **START** above to switch on the camera.")

    render_status()

    st.button(
        "🗑️ Clear Message",
        use_container_width=True,
        on_click=lambda: ctx.video_processor.request_clear() if ctx.video_processor else None,
    )

    with st.expander("ℹ️ How to use"):
        st.markdown(
            "- Allow camera access when prompted.\n"
            "- Look at the camera and blink deliberately.\n"
            "- A **short blink** records a dot, a **long blink** records a dash.\n"
            "- Keep your eyes **open** for the letter-pause duration to decode a letter.\n"
            "- Keep them open a little longer to insert a space between words.\n"
            "- Use **Clear Message** to start over."
        )

# ----------------------------------------------------------------------
# 6. PUSH SETTINGS
#    Runs once per full-page rerun (e.g. whenever a sidebar slider
#    changes), independently of the status fragment above.
# ----------------------------------------------------------------------
if ctx.video_processor:
    ctx.video_processor.update_settings(
        ear_threshold, long_blink_threshold, letter_pause, word_pause, show_mesh
    )