import streamlit as st
import trio
from trio import SocketStream
import sys

st.markdown("# Test app")

input_text = st.text_input(label="Enter some text 👇")


async def send_info(client_stream:SocketStream, 
                    input_text:str) -> None:
    await client_stream.send_all(input_text.encode('utf-8'))


async def receive_info(client_stream:SocketStream) -> None:
    async for data in client_stream:
        st.write(f"Received data -> {data}")
    sys.exit()
    


PORT = 12345
HOST = "192.168.1.3"

async def main() -> None:
    
    global input_text

    if "main_session" not in st.session_state:
        st.session_state['main_session'] = True

    client_stream = await trio.open_tcp_stream(HOST, PORT)
    async with client_stream:
        async with trio.open_nursery() as nursery:
            if st.session_state['main_session']:
                nursery.start_soon(receive_info, client_stream)
            if st.button(label="Send info"):
                nursery.start_soon(send_info, *(client_stream, input_text))
                st.session_state['main_session'] = False

trio.run(main)







