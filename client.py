import sys
import trio

PORT = 12345

async def sender(client_stream):
    print("Sender started")
    while True:
        data = b"Test message"
        print(f"Sender sending {data}")
        await client_stream.send_all(data)
        await trio.sleep(1)


async def receiver(client_stream):
    print("Receiver started")
    async for data in client_stream:
        print(f"Receiver got data: {data}")
    print("Receiver connection closed")
    sys.exit()


async def main():
    print(f"Connecting to port {PORT}")
    client_stream = await trio.open_tcp_stream("192.168.1.3", PORT)
    async with client_stream:
        async with trio.open_nursery() as nursery:
            print("Parent spawning sender")
            nursery.start_soon(sender, client_stream)

            print("Parent spawning receiver")
            nursery.start_soon(receiver, client_stream)



if __name__ == "__main__":
    trio.run(main)