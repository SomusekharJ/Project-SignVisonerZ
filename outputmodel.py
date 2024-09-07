import pickle
import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime, timedelta

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize Mediapipe for hand detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Labels dictionary for 26 letters
labels_dict = {i: chr(65 + i) for i in range(26)}  

# Initialize variables for letter and sentence tracking
predicted_character = ""
sentence = ""
hand_visible_start_time = None

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape

    # Convert the image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and detect hands
    results = hands.process(frame_rgb)
    hand_visible = False

    if results.multi_hand_landmarks:
        hand_visible = True
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Collect landmark data
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Pad or trim data_aux to the required length (84 features)
        data_aux_padded = np.pad(data_aux, (0, 84 - len(data_aux)), mode='constant') if len(data_aux) < 84 else data_aux[:84]

        # Make prediction using the model
        prediction = model.predict([data_aux_padded])
        predicted_character = labels_dict[int(prediction[0])]

        # Update the sentence if hand is visible for 3 seconds
        if hand_visible:
            if hand_visible_start_time is None:
                hand_visible_start_time = datetime.now()
            elif (datetime.now() - hand_visible_start_time) > timedelta(seconds=3):
                sentence += predicted_character
                hand_visible_start_time = None  # Reset the timer
        else:
            hand_visible_start_time = None  # Reset the timer if hand is not visible

    # Display sections on the frame
    cv2.putText(frame, f'Converted sign text: {predicted_character}', (10, H - 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, f'Sentence: {sentence}', (10, H - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()