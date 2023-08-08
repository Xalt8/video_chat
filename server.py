import trio
import socket

# - must be in between 1024 and 65535
# - can't be in use by some other program on your computer
# - must match what we set in our echo server
PORT = 12345
HOST = socket.gethostbyname(socket.gethostname()) 



async def server_handler(server_stream):
    try:
        async for data in server_stream:
            print(f"Received data -> {data}")
            await server_stream.send_all(data)
            print(f"Sent data -> {data}")
    except Exception as e:
        print("Exception in server handler", e)


async def main():
    try:
        await trio.serve_tcp(handler=server_handler, port=PORT, host=HOST)
    except Exception as e:
        print("Exception in main", e)





if __name__ == "__main__":
    print("server module")
    trio.run(main)