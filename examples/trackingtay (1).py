import cv2
import mediapipe as mp
import numpy as np

# Constants
MIN_SCROLL_THRESHOLD = 0.01
PINCH_THRESHOLD = 0.035

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Initialize MediaPipe drawing
mp_drawing = mp.solutions.drawing_utils

def main():
    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    pinch_detected = False
    thumb_tip_prev = None
    scroll_detected = False

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            distance = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_tip.x, index_tip.y]))

            if distance < PINCH_THRESHOLD:
                if not pinch_detected:
                    pinch_detected = True
                    thumb_tip_prev = thumb_tip.y
                    
                else:
                    vertical_movement = thumb_tip.y - thumb_tip_prev
                    if abs(vertical_movement) > MIN_SCROLL_THRESHOLD:
                        scroll_amount = -vertical_movement * 400  # Negative to invert direction
                        print(f'Scroll amount: {scroll_amount:.2f}')
                        scroll_detected = True
                    thumb_tip_prev = thumb_tip.y
            elif pinch_detected:
                pinch_detected = False
                if scroll_detected:
                    scroll_detected = False
                else:
                    print("Pinch")

        cv2.imshow('Hand Tracking', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()