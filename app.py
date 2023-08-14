import streamlit as st
import trio
from trio import SocketStream, MemoryReceiveChannel, MemorySendChannel
import cv2
import numpy as np
from collections.abc import Iterator


st.title("Test app")

PORT = 12345
HOST = "192.168.1.3"


async def send_info(client_stream:SocketStream, 
                    input_text:str) -> None:
    await client_stream.send_all(input_text.encode('utf-8'))


async def receive_info(client_stream:SocketStream) -> None:
    async for data in client_stream:
        st.write(f"Received data -> {data}")


async def send_video_frames(client_stream:SocketStream, 
                            frame:np.ndarray, 
                            encoder:str='.jpg') -> None:
    """ Takes an video frame, encodes it and sends it using a socket"""
    encoding_sucess, encoded_frame = cv2.imencode(encoder, frame)
    if encoding_sucess:
        frame_encoded_bytes = encoded_frame.tobytes()
        await client_stream.send_all(frame_encoded_bytes)
    else:
        print("Something went wrong with video encoding")
        return


async def receive_video_frames(client_stream:SocketStream, 
                               send_channel:MemorySendChannel) -> None:
    """ Recieves video from from socket, decodes it as an array """
    async with send_channel:
        async for frame_encoded_bytes in client_stream:
            arr_from_bytes = np.frombuffer(frame_encoded_bytes, dtype=np.uint8)
            decoded_frame = cv2.imdecode(arr_from_bytes, cv2.IMREAD_COLOR)
            await send_channel.send(decoded_frame)


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


async def stream_video() -> None:
    frame_place_holders = st.empty()
    col1, col2 = st.columns([1,1])
    with col1:
        start_button_pressed = st.button(label="Start")
    with col2:
        stop_button_pressed = st.button(label="Stop")
    
    client_stream = await trio.open_tcp_stream(HOST, PORT)
    async with client_stream:
        if start_button_pressed:
            for frame in capture_video_frames():
                async with trio.open_nursery() as nursery:
                    send_channel, receive_channel = trio.open_memory_channel(max_buffer_size=0)
                    nursery.start_soon(send_video_frames, client_stream, frame)
                    nursery.start_soon(receive_video_frames, client_stream, send_channel)

                    # Get the frame from receive_video_frames()
                    received_frame = await receive_channel.receive() 

                    # Display the frames in the app
                    frame_place_holders.image(received_frame)
                
                    # if cv2.waitKey(1) & 0xff == ord('q') or stop_button_pressed:
                    #     break
                if stop_button_pressed:
                    break
    
    
trio.run(stream_video)
    
    
