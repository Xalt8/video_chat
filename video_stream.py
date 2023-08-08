import cv2


def capture_webcam_video() -> None:
    """ Uses opencv to capture the video from webcam and display it
        in a window """
    
    # Change webcam_num from 1 to 0 to switch front/back
    webcam_num = 1

    vid = cv2.VideoCapture(webcam_num)

    VIDEO_WIDTH = 640
    VIDEO_HEIGHT = 480

    vid.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
    
    while True:
        _, frame = vid.read()
        
        resized_frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))

        cv2.putText(img=resized_frame, 
                    text="Press 'q' to quit", 
                    org=(VIDEO_WIDTH-400, VIDEO_HEIGHT-10), 
                    fontFace=5, 
                    fontScale=1, 
                    color=(255,255,255), 
                    thickness=2)


        cv2.imshow('frame', resized_frame)
        
        # Use the 'q' key to break the video capture
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_webcam_video()