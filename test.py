import cv2
import numpy as np
import struct

if __name__== "__main__":

    img = cv2.imread("pic.jpg")
    img = cv2.resize(src=img, dsize=(640,480))

    # Sending image
    encoding_sucess, encoded_img = cv2.imencode('.jpg', img)
    assert encoding_sucess, "Encoding error"
    img_encoded_bytes = encoded_img.tobytes()
    frame_size = len(img_encoded_bytes)
    print(f"frame_size: {frame_size}")
    header = struct.pack('!I', frame_size)
    print(f"header: {header!r}")
    packet = header + img_encoded_bytes


    # Receiving image
    extracted_header = packet[:4]
    print(f"extracted_header: {extracted_header!r}")
    frame_size_received = struct.unpack("!I", header)[0]
    print(f"frame_size_received {frame_size_received}")

    arr_from_buffer:np.ndarray = np.frombuffer(img_encoded_bytes, dtype=np.uint8)
    decoded_img = cv2.imdecode(arr_from_buffer, cv2.IMREAD_COLOR)
    print("Ok")
    
    # print(img.all() == decoded_img.all())

    # cv2.imshow('decoded_img', decoded_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()