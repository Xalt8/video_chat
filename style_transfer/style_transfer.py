import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path


# Source: https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/examples/style_transfer/overview.ipynb#scrollTo=Cg0Vi-rXRUFl
# Model -> https://tfhub.dev/google/lite-model/magenta/arbitrary-image-stylization-v1-256/int8/transfer/1

STYLE_PREDICTION_PATH:str = Path("magenta_arbitrary-image-stylization-v1-256_int8_prediction_1.tflite").as_posix()
STYLE_TRANSFORM_PATH:str = Path("magenta_arbitrary-image-stylization-v1-256_int8_transfer_1.tflite").as_posix()

style_image_folder = Path("images/style_images") 
content_image_folder = Path("images/content_images")


def load_image(image_path:str) -> np.ndarray:
    """ Takes an image path and returns an RGB image array """
    img = cv2.imread(filename= image_path, flags= cv2.IMREAD_COLOR) # <- BGR colour channels 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # <- unit8
    return img

def pre_process_image(image:np.ndarray, shape:tuple[int, int]) -> np.ndarray:
    img = cv2.resize(src=image, dsize=shape, interpolation=cv2.INTER_NEAREST)
    img = (img / 255.0).astype(np.float32) # <- normalize values between [0,1] 
    img = np.expand_dims(a=img, axis=0) # <- add batch dimension
    return img    


def run_style_predict(processed_style_image:np.ndarray) -> np.ndarray:
    """ Takes in an image an returns a style bottleneck """

    assert processed_style_image.shape == (1, 256, 256, 3), "Image shape should be (1, 256, 256, 3)"
    intepreter = tf.lite.Interpreter(model_path= STYLE_PREDICTION_PATH)
    intepreter.allocate_tensors()
    input_details = intepreter.get_input_details()
    intepreter.set_tensor(input_details[0]['index'], processed_style_image)
    intepreter.invoke()
    style_bottleneck = intepreter.tensor(intepreter.get_output_details()[0]["index"])()
    return style_bottleneck


def run_style_transform(style_bottleneck:np.ndarray, 
                        processed_content_image:np.ndarray) -> np.ndarray:
    """ Takes a style bottleneck and a content image an returns a stylized image """

    assert processed_content_image.shape == (1,384,384,3), "Image shape should be (1,384,384,3)"
    intepreter = tf.lite.Interpreter(model_path= STYLE_TRANSFORM_PATH)
    intepreter.allocate_tensors()
    input_details = intepreter.get_input_details()
    intepreter.set_tensor(input_details[0]["index"], processed_content_image)
    intepreter.set_tensor(input_details[1]["index"], style_bottleneck)
    intepreter.invoke()
    stylized_image = intepreter.tensor(intepreter.get_output_details()[0]["index"])()
    return stylized_image
    

def blend_style(content_image:np.ndarray,
                style_image:np.ndarray,
                content_blending_ratio:float=0.50) -> np.ndarray:
        """ Blend the style of the content image into the stylized image 
            0-> less style from content image, 1-> more style from content image """
        
        assert 0 <= content_blending_ratio <= 1, "content_blending_ratio should be between 0 & 1"
        processed_content_image_blend = pre_process_image(image = content_image, shape=(256,256))
        processed_content_image = pre_process_image(image=content_image, shape=(384,384))
        style_bottleneck_content = run_style_predict(processed_style_image= processed_content_image_blend)
        
        processed_style_image = pre_process_image(image=style_image, shape=(256,256))
        style_bottleneck = run_style_predict(processed_style_image= processed_style_image)

        style_bottleneck_blended = content_blending_ratio * style_bottleneck_content + (1 - content_blending_ratio) * style_bottleneck
        stylized_image_blended = run_style_transform(style_bottleneck=style_bottleneck_blended,
                                                 processed_content_image=processed_content_image)
        return stylized_image_blended


if __name__ == "__main__":
    
    content_image = load_image(image_path= (content_image_folder/"in_bruges.png").as_posix())
    
    style_images = [load_image(image_path= (style_image_folder/style_image.name).as_posix()) 
                    for style_image in style_image_folder.iterdir()]

    fig, axs = plt.subplots(nrows=len(style_images), ncols=5, figsize=(8,6))

    for row, style_image in enumerate(style_images):
        axs[row,0].imshow(content_image)
        axs[row,0].set_title('Content Image')
        axs[row,0].axis('off')

        axs[row,1].imshow(style_image)
        axs[row,1].set_title('Style Image')
        axs[row,1].axis('off')

        for i, perc in enumerate([y/4 for y in range(3)], 2):
            axs[row, i].imshow(np.squeeze(blend_style(content_image=content_image,
                                                    style_image=style_image,
                                                    content_blending_ratio=perc)))
            axs[row, i].set_title(f'{perc*100:g}% content')
            axs[row, i].axis('off')    
        
    plt.tight_layout()
    plt.show()




