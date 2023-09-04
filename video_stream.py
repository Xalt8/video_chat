import cv2
import numpy as np
from collections.abc import Iterator
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
from background_remove import get_segmentation_map


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
    
    segmentor = SelfiSegmentation()

    while vid.isOpened():
        _, frame = vid.read()
        resized_frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
        cv2.putText(img=resized_frame, 
                    text="Press 'q' to quit", 
                    org=(VIDEO_WIDTH-400, VIDEO_HEIGHT-10), 
                    fontFace=5, 
                    fontScale=1, 
                    color=(255,255,255), 
                    thickness=2)

        img_out = segmentor.removeBG(img=resized_frame, imgBg=(255,0,255), threshold=0.1)

        img_stacked = cvzone.stackImages([resized_frame, img_out], cols=2, scale=1)
        cv2.imshow('stacked images', img_stacked)

        # cv2.imshow('frame', resized_frame)
        # cv2.imshow('img_out', img_out)
        # Use the 'q' key to break the video capture
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()


def capture_video_frames(frame_width:int=640, frame_height:int=480, cam_num:int=1) -> Iterator[np.ndarray]:
    """ Captures video frames using webcam and returns it as an array """
    vid = cv2.VideoCapture(cam_num, cv2.CAP_DSHOW)
    if vid.isOpened():
        vid.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        vid.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    try:
        while True:
            ret, frame = vid.read()
            if not ret:
                print("Video feed ended")
                break
            resized_frame = cv2.resize(src=frame, dsize=(frame_width, frame_height))
            recoloured_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            yield recoloured_frame
    finally:
        vid.release()
        cv2.destroyAllWindows()   



def capture_webcam_video2(frame_width:int=640, frame_height:int=480, cam_num:int=0) -> None:
    """ Uses opencv to capture the video from webcam and display it
        in a window """
    vid = cv2.VideoCapture(cam_num, cv2.CAP_DSHOW)
    if vid.isOpened():
        vid.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        vid.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    try:    
        while True:
            _, frame = vid.read()
            resized_frame = cv2.resize(frame, (frame_width, frame_height))
            cv2.putText(img=resized_frame, 
                        text="Press 'q' to quit", 
                        org=(frame_width-400, frame_height-10), fontFace=5, fontScale=1, 
                        color=(255,255,255), thickness=2)
            # Get a seg map of image
            seg_map = get_segmentation_map(model_path='lite-model_mobilenetv2-dm05-coco_fp16_1.tflite', image=resized_frame)
            seg_map = np.expand_dims(seg_map, axis=2)
            # Make background white
            white_bg = np.where((seg_map==0), [255,255,255], resized_frame).astype(np.uint8)
            
            # img_stacked = cvzone.stackImages([resized_frame, white_bg], cols=2, scale=1)
            cv2.imshow('white_bg', white_bg)

            # Use the 'q' key to break the video capture
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
    finally:
        vid.release()
        cv2.destroyAllWindows()


          


if __name__ == "__main__":
    capture_webcam_video()