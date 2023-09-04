import streamlit as st
import cv2
import asyncio
from asyncio import StreamReader, StreamWriter
import struct
import numpy as np
from collections.abc import Iterator
from video_stream import capture_video_frames

st.title("Test app")

if 'start_streaming' not in st.session_state:
    st.session_state['start_streaming'] = False


def start_button_pressed():
    st.session_state['start_streaming'] = True
       

def stop_button_pressed():
    st.session_state['start_streaming'] = False


frame_place_holders = st.empty()
col1, col2 = st.columns([1,1])
with col1:
    st.button(label="Start", on_click=start_button_pressed)
with col2:
    st.button(label="Stop", on_click=stop_button_pressed)


async def pack_and_write(frame:np.ndarray, writer:StreamWriter) -> None:
    # Takes a video frame and writes to server
    encoding_sucess, encoded_frame = cv2.imencode('.jpg', frame)
    assert encoding_sucess, "Encoding error"
    frame_encoded_bytes = encoded_frame.tobytes()
    frame_size = len(frame_encoded_bytes)
    header = struct.pack('!I', frame_size)
    packet = header + frame_encoded_bytes
    writer.write(packet)
    await writer.drain()


async def read_and_unpack(reader:StreamReader) -> np.ndarray:
    header = await reader.readexactly(4) # Get the first 4 bytes
    frame_size = struct.unpack("!I", header)[0]
    frame_encoded_bytes = await reader.readexactly(frame_size)
    arr_from_bytes = np.frombuffer(frame_encoded_bytes, dtype=np.uint8)
    decoded_frame = cv2.imdecode(arr_from_bytes, cv2.IMREAD_COLOR)
    return decoded_frame


async def main():
    PORT = 12345
    HOST = "192.168.1.3"
    
    reader, writer = await asyncio.open_connection(HOST, PORT)
    
    while st.session_state['start_streaming'] == True:
        for frame in capture_video_frames(cam_num=0):
            # Send the frame to server
            await pack_and_write(frame=frame, writer=writer)
            
            # Receive the frame from server
            received_frame = await read_and_unpack(reader=reader)

            # Display the received frame from server on app 
            assert received_frame.shape == frame.shape, "received_frame and frame are not the same shape"
            frame_place_holders.image(received_frame)

            if st.session_state['start_streaming'] == False:
                writer.close()
                await writer.wait_closed() 


asyncio.run(main())
