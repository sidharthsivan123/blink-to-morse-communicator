import time
import mediapipe as mp
import numpy as np
import cv2
# 1. MEDIAPIPE FACE MESH SETUP
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
# 2. EYE LANDMARKS
RIGHT_EYE = [33, 159, 158, 133, 153, 145]
LEFT_EYE = [362, 386, 385, 263, 380, 374]
# 3. EAR CALCULATION
def calculate_ear(eye_points):
    vertical_1 = np.linalg.norm(
        eye_points[1] - eye_points[5]
    )
    vertical_2 = np.linalg.norm(
        eye_points[2] - eye_points[4]
    )
    horizontal = np.linalg.norm(
        eye_points[0] - eye_points[3]
    )
    ear = (
        vertical_1 + vertical_2
    ) / (2.0 * horizontal)
    return ear
#4.SETTINGS
EAR_THRESHOLD = 0.20
LONG_BLINK_THRESHOLD = 0.60
LETTER_PAUSE = 1.20
WORD_PAUSE = 2.50
#5.VARIABLES
eyes_closed = False
blink_start_time = None
last_blink_time = None
last_letter_time = None
current_morse = ""
decoded_message = ""
#6.MORSE CODE DICTIONARY
MORSE_CODE = {'A': '.-','B': '-...','C': '-.-.','D': '-..','E': '.','F': '..-.','G': '--.',
    'H': '....','I': '..','J': '.---','K': '-.-','L': '.-..','M': '--','N': '-.','O': '---',
    'P': '.--.','Q': '--.-','R': '.-.','S': '...','T': '-','U': '..-','V': '...-',
    'W': '.--','X': '-..-','Y': '-.--','Z': '--..',
    '0': '-----','1': '.----','2': '..---',
    '3': '...--','4': '....-',
    '5': '.....',
    '6': '-....',
    '7': '--...',
    '8': '---..',
    '9': '----.'
}
MORSE_TO_TEXT = {
    value: key
    for key, value in MORSE_CODE.items()
}
#7.CLEAR FUNCTION
def clear_message():
    global decoded_message
    global current_morse
    global last_blink_time
    global last_letter_time
    global blink_start_time
    global eyes_closed
    decoded_message = ""
    current_morse = ""
    last_blink_time = None
    last_letter_time = None
    blink_start_time = None
    eyes_closed = False
    print("Message cleared")
#8.MOUSE CALLBACK FOR CLEAR BUTTON
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Clear button coordinates
        if (
            500 <= x <= 620
            and 20 <= y <= 60
        ):
            clear_message()
# 9. WEBCAM
video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Error: Could not access webcam.")
else:
    # Create window
    cv2.namedWindow(
        "Blink to Morse Communicator"
    )
    # Connect mouse callback
    cv2.setMouseCallback(
        "Blink to Morse Communicator",
        mouse_callback
    )
    # 10. MAIN LOOP
    while True:
        suc, img = video.read()
        if not suc:
            print("Failed to read webcam.")
            break
        # Mirror webcam
        img = cv2.flip(img, 1)
        # Get image dimensions
        h, w, _ = img.shape
        # Convert BGR to RGB
        img1 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB
        )
        # Process image using MediaPipe
        result = face_mesh.process(img1)
        # Default eye status
        eye_status = "NO FACE"
        # 11. FACE DETECTION
        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                # Draw Face Mesh
                mp_draw.draw_landmarks(img,face_landmarks,mp_face_mesh.FACEMESH_TESSELATION,
                    mp_draw.DrawingSpec(color=(0, 0, 255),thickness=1,circle_radius=1),
                    mp_draw.DrawingSpec(color=(0, 0, 0),thickness=1
                    )
                )
                # Get Eye Points
                right_eye_points = []
                left_eye_points = []
                # Right Eye
                for index in RIGHT_EYE:
                    landmark = face_landmarks.landmark[index]
                    px = int(
                        landmark.x * w
                    )
                    py = int(
                        landmark.y * h
                    )
                    right_eye_points.append(
                        np.array([px, py])
                    )
                    cv2.circle(img,(px, py),3,(0, 255, 0),-1
                    )
                #Left Eye
                for index in LEFT_EYE:
                    landmark = face_landmarks.landmark[index]
                    px = int(
                        landmark.x * w
                    )
                    py = int(
                        landmark.y * h
                    )
                    left_eye_points.append(
                        np.array([px, py])
                    )
                    cv2.circle(img,(px, py),3,(0, 255, 0),-1
                    )
                #Convert Eye Points to NumPy
                right_eye_points = np.array(
                    right_eye_points
                )
                left_eye_points = np.array(
                    left_eye_points
                )
                # Calculate EAR
                right_ear = calculate_ear(
                    right_eye_points
                )
                left_ear = calculate_ear(
                    left_eye_points
                )
                # Average EAR
                ear = (
                    right_ear + left_ear
                ) / 2
                # Detect Eye State
                if ear < EAR_THRESHOLD:
                    eye_status = "CLOSED"
                   # Start Blink
                    if not eyes_closed:
                        eyes_closed = True
                        blink_start_time = time.time()
                else:
                    eye_status = "OPEN"
                    # Blink Completed
                    if eyes_closed:
                        eyes_closed = False
                        # Calculate blink duration
                        blink_duration = (
                            time.time()
                            - blink_start_time
                        )
                        # Convert Blink to Morse
                        if (
                            blink_duration
                            < LONG_BLINK_THRESHOLD
                        ):
                            current_morse += "."
                            symbol = "."
                        else:
                            current_morse += "-"
                            symbol = "-"
                        print(
                            f"Blink: "
                            f"{blink_duration:.2f}s"
                        )
                        print(
                            f"Symbol: {symbol}"
                        )
                        print(
                            f"Current Morse: "
                            f"{current_morse}"
                        )
                        # Record blink end time
                        last_blink_time = time.time()
                        # Reset blink timer
                        blink_start_time = None
                # LETTER PAUSE
                if (
                    not eyes_closed
                    and last_blink_time is not None
                ):
                    pause = (
                        time.time()
                        - last_blink_time
                    )
                    if pause >= LETTER_PAUSE:
                        if current_morse != "":
                            # Check Morse Code
                            if (
                                current_morse
                                in MORSE_TO_TEXT
                            ):
                                letter = (
                                    MORSE_TO_TEXT[
                                        current_morse
                                    ]
                                )
                                # Add letter
                                decoded_message += letter
                                # Record letter time
                                last_letter_time = (
                                    time.time()
                                )
                                print(
                                    f"Letter: "
                                    f"{letter}"
                                )
                                print(
                                    f"Message: "
                                    f"{decoded_message}"
                                )
                            else:
                                print(
                                    "Unknown Morse:",
                                    current_morse
                                )
                            # Reset Morse
                            current_morse = ""
                        # Reset blink time
                        last_blink_time = None
                # WORD PAUSE
                if (
                    last_letter_time is not None
                    and last_blink_time is None
                    and not eyes_closed
                ):
                    word_pause = (
                        time.time()
                        - last_letter_time
                    )
                    if word_pause >= WORD_PAUSE:
                        if (
                            decoded_message != ""
                            and
                            not decoded_message.endswith(" ")
                        ):
                            # Add space
                            decoded_message += " "
                            print(
                                "Word completed"
                            )
                            print(
                                f"Message: "
                                f"{decoded_message}"
                            )
                        last_letter_time = None
                # DISPLAY EAR
                cv2.putText(
                    img,
                    f"EAR: {ear:.3f}",
                    (30, 40),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0, 255, 0),2)
                # DISPLAY EYE STATUS
                cv2.putText(
                    img,
                    f"Eyes: {eye_status}",
                    (30, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )
                # DISPLAY MORSE
                cv2.putText(
                    img,
                    f"Code: {current_morse}",(30, 110),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255, 255, 0),2)
                # DISPLAY MESSAGE
                cv2.putText(
                    img,
                    f"Message: {decoded_message}",
                    (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )
        #12.CLEAR BUTTON
        cv2.rectangle(
            img,
            (500, 20),
            (620, 60),
            (0, 0, 255),
            -1
        )
        cv2.putText(
            img,
            "CLEAR",
            (520, 47),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        #13.SHOW WINDOW    
        cv2.imshow(
            "Blink to Morse Communicator",
            img
        )
        #14.PRESS Q TO EXIT
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
# 15. RELEASE RESOURCES
video.release()
cv2.destroyAllWindows()