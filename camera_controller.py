import cv2


def camera_capture(output_size=(224, 224), normalize=True, video_channel=0):
    cap = cv2.VideoCapture(video_channel)
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, output_size)
    if normalize:
        frame = frame.astype('float32')
        frame = frame / 255.0
    return frame
