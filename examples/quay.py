import cv2
import time

def record_cameras():
    eye_cam = cv2.VideoCapture(2)  # Adjust index if necessary
    front_cam = cv2.VideoCapture(1)  # Adjust index if necessary
    front_cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    if not eye_cam.isOpened() or not front_cam.isOpened():
        print("Failed to open one or both cameras")
        return

    # Set camera resolutions
    eye_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    eye_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    front_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    front_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Define the codec and create VideoWriter objects
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    eye_out = cv2.VideoWriter('eye_output.avi', fourcc, 30.0, (320, 240))
    front_out = cv2.VideoWriter('front_output.avi', fourcc, 30.0, (640, 480))

    start_time = time.time()
    duration = 45  # Record for 10 seconds

    while True:
        ret_eye, eye_frame = eye_cam.read()
        ret_front, front_frame = front_cam.read()

        if not ret_eye or not ret_front:
            print("Failed to grab frame from one of the cameras")
            break

        eye_out.write(eye_frame)
        front_out.write(front_frame)

        cv2.imshow("Eye Camera", eye_frame)
        cv2.imshow("Front Camera", front_frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or time.time() - start_time > duration:
            break

    eye_cam.release()
    front_cam.release()
    eye_out.release()
    front_out.release()
    cv2.destroyAllWindows()

    print("Recording complete. Output saved as 'eye_output.avi' and 'front_output.avi'")

if __name__ == "__main__":
    record_cameras()