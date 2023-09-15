import cv2
import mediapipe as mp
from mediapipe.python.solutions.face_mesh import FaceMesh
from mediapipe.tasks import python
from typing import Iterator
import numpy as np
import matplotlib.pyplot as plt
import dlib


def get_video_frames(video_path:str) -> Iterator[np.ndarray]:
    """ Takes a path to a video location and returns the video frames """
    vid = cv2.VideoCapture(video_path)
    try:
        while True:
            sucess, frame = vid.read()
            if not sucess:
                break
            yield frame
    finally:
        vid.release()
        cv2.destroyAllWindows()    


def get_landmarks_dlib(video_path:str, model_path:str) -> None:
    hog_face_detector = dlib.get_frontal_face_detector()
    face_landmark_predictor = dlib.shape_predictor(model_path)
    # Boundry box around the face
    for frame in get_video_frames(video_path):
        faces = hog_face_detector(frame)
        for face in faces:
            cv2.rectangle(img=frame, 
                        pt1=(face.left(), face.top()), pt2=(face.right(), face.bottom()), 
                        color=(0,255,0), thickness=3)
            # Get face landmarks
            face_landmarks = face_landmark_predictor(frame, face)
            for n in range(0, 68):
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y
                cv2.circle(img=frame, center=(x,y), radius=2, color=(0,255,0), thickness=-1)
        cv2.imshow("video", frame)
        cv2.waitKey(1)


def get_landmarks_facemesh2(video_path:str, model_path:str) -> None:
    """ This function requires a model to be downloaded to the system 
        The image frame needs to be converted to a media pipe Image format"""
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = python.vision.FaceLandmarkerOptions(base_options=base_options,
                                                  output_face_blendshapes=True,
                                                  output_facial_transformation_matrixes=True,
                                                  num_faces=1)
    detector = python.vision.FaceLandmarker.create_from_options(options)
    
    for frame in get_video_frames(video_path):
        frame_height, frame_width, _ = frame.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = detector.detect(mp_image)
        faces = detection_result.face_landmarks # Detects faces
        for i in range(len(faces)): # Loop through the faces
            face_landmarks = faces[i]
        for j in range(0, len(face_landmarks)): # Loop through the number of keypoints 
            x = int(face_landmarks[j].x * frame_width)
            y = int(face_landmarks[j].y * frame_height)
                
            cv2.circle(img = frame, center=(x,y), radius=1, color=(0,255,255), thickness=-1)

        cv2.imshow("video", frame)
        cv2.waitKey(1)


def get_landmarks_facemesh(video_path:str) -> None:
    ''' Source: https://www.youtube.com/watch?v=LGPBRH6Hqw8'''
    for frame in get_video_frames(video_path):
        img_height, img_width, _ = frame.shape

        face_lips_connections = mp.solutions.face_mesh.FACEMESH_LIPS
        face_lip_indicies = [item for tups in list(face_lips_connections) for item in tups]

        results = FaceMesh(static_image_mode=True).process(frame)
        for facial_landmarks in results.multi_face_landmarks:
            for index in face_lip_indicies:
                x = int(facial_landmarks.landmark[index].x * img_width)
                y = int(facial_landmarks.landmark[index].y * img_height)
    
                cv2.circle(img=frame, center=(x,y), radius=2, color=(0,255,0), thickness=-1)    
        
        cv2.imshow("video", frame)
        cv2.waitKey(1)



if __name__ == "__main__":
    
    
    get_landmarks_facemesh(video_path="woman_vid.mp4")
    # get_landmarks_dlib(video_path="woman_vid.mp4", model_path="shape_predictor_68_face_landmarks.dat")