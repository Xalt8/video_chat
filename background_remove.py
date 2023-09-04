import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def prep_for_prediction(image:np.ndarray, input_size:tuple[int, int]) -> np.ndarray:
    """ Takes an image resizes it, adds an batch dimension and normalizes
        values between 0 & 1 """
    image = cv2.resize(src=image, dsize=input_size, interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32)
    image = np.expand_dims(a=image, axis=0)
    image = image / 127.5 - 1
    return image


def get_segmentation_map(model_path:str, image:np.ndarray) -> np.ndarray:
    """ Takes an path to model, creates a model interpreter, passes the image
        through the model for segmentation 
        Source: https://tfhub.dev/sayakpaul/lite-model/deeplabv3-mobilenetv2/1/default/1
    """
    interpreter = tf.lite.Interpreter(model_path = model_path) # Load the model
    # Invoke the interpreter
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    input_size = input_shape[2], input_shape[1] #(513, 513)
    interpreter.allocate_tensors()
    # Get the image ready for prediction
    image_for_pred = prep_for_prediction(image=image, input_size=input_size)
    interpreter.set_tensor(tensor_index=input_details[0]['index'], value=image_for_pred)
    interpreter.invoke() # Forward pass on model 
    # Get raw output map 
    raw_pred = interpreter.tensor(interpreter.get_output_details()[0]['index'])()
    # Post processing
    seg_map = np.squeeze(tf.argmax(input=raw_pred, axis=3)).astype(np.float32)
    height, width = image.shape[:2] # Original image's size
    seg_map_resized = cv2.resize(src=seg_map, dsize=(width, height), interpolation=cv2.INTER_NEAREST)
    return seg_map_resized.astype(np.uint8)

LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'])
    


if __name__ == "__main__":
    img1 = cv2.imread('zoom_pic.jpg', cv2.IMREAD_COLOR) # <- BGR colour channels 
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    seg_map = get_segmentation_map(model_path='lite-model_mobilenetv2-dm05-coco_fp16_1.tflite', image=img1)
    seg_map = np.expand_dims(seg_map, axis=2)

    white_bg = np.where((seg_map==0), [255,255,255], img1)
    remove_human = np.where((seg_map==15), [255,0,0], img1)
    remove_non_human = np.where((seg_map!=15), [255,255,255], img1)
    
    # 2nd image
    img2 = cv2.imread('woman_cat.jpg', cv2.IMREAD_COLOR) # <- BGR colour channels 
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    seg_map2 = get_segmentation_map(model_path='lite-model_mobilenetv2-dm05-coco_fp16_1.tflite', image=img2)
    seg_map2 = np.expand_dims(seg_map2, axis=2)

    white_bg2 = np.where((seg_map2==0), [255,255,255], img2)
    remove_human2 = np.where((seg_map2==15), [255,0,0], img2)
    not_bg_not_human = np.where((seg_map2!=0) & (seg_map2!=15) , [255,0,0], img2)
    

    #Plot the results
    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(10, 8))
    
    axs[0,0].imshow(img1)
    axs[0,0].set_title('Image')
    axs[0,0].axis('off')

    axs[0,1].imshow(seg_map)
    axs[0,1].set_title("Segmentation map")
    axs[0,1].axis('off')

    axs[0,2].imshow(img1)
    axs[0,2].imshow(seg_map, alpha=0.5)
    axs[0,2].set_title("Overlaid")
    axs[0,2].axis('off')

    axs[1,0].imshow(white_bg)
    axs[1,0].set_title("White background")
    axs[1,0].axis('off')

    axs[1,1].imshow(remove_human)
    axs[1,1].set_title("Remove human")
    axs[1,1].axis('off')

    axs[1,2].imshow(remove_non_human)
    axs[1,2].set_title("Remove non human")
    axs[1,2].axis('off')

    axs[2,0].imshow(img2)
    axs[2,0].set_title("Cat, human & background")
    axs[2,0].axis('off')

    axs[2,1].imshow(seg_map2)
    axs[2,1].set_title("Segmentation map2")
    axs[2,1].axis('off')

    axs[2,2].imshow(img2)
    axs[2,2].imshow(seg_map2, alpha=0.5)
    axs[2,2].set_title("Overlaid")
    axs[2,2].axis('off')

    axs[3,0].imshow(white_bg2)
    axs[3,0].set_title("White background")
    axs[3,0].axis('off')

    axs[3,1].imshow(remove_human2)
    axs[3,1].set_title("Remove human")
    axs[3,1].axis('off')

    axs[3,2].imshow(not_bg_not_human)
    axs[3,2].set_title("Remove anything not human\n& not background")
    axs[3,2].axis('off')
    
    plt.tight_layout()
    
