import cv2
import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path


def resize_image(image:np.ndarray, target_shape:tuple) -> np.ndarray:
    """ Takes an input image and resizes it by a target shape"""
    assert image.ndim == 3, "Image should be in the shape (width,height,channels)"
    original_height, original_width = image.shape[:2]
    target_height, target_width = target_shape[:2]
    original_aspect_ratio = original_width/original_height 

    if original_aspect_ratio > 1:
        new_width = target_width
        new_height = int(target_width / original_aspect_ratio)
    else:
        new_height = target_height
        new_width = int(target_height * original_aspect_ratio)
    
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Create a blank image in the shape of the target image
    final_image = np.zeros(shape=target_shape, dtype=np.uint8)

    # Where to put the image
    put_x = (target_width - new_width) // 2
    put_y = (target_height - new_height) // 2

    final_image[put_y:put_y + new_height, put_x:put_x + new_width] = resized_image
    
    return final_image


def display_image(image:np.ndarray, title:str=':)') -> None:
    title = f"{image=}".split("=")[0] if title == ':)' else title
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__== "__main__":
    
    # img = cv2.imread('zoom_pic.jpg', cv2.IMREAD_COLOR) # <- BGR colour channels 
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # seg_map = np.load('seg_map.npy')
    # seg_map_3d = np.expand_dims(seg_map, axis=2)

    # white_bg = np.where((seg_map_3d==0), [255,255,255], img)
    # remove_human = np.where((seg_map_3d==15), [255,0,0], img)
    # remove_non_human = np.where((seg_map_3d!=15), [255,255,255], img)

    # # 2nd image
    # img2 = cv2.imread('woman_cat.jpg', cv2.IMREAD_COLOR) # <- BGR colour channels 
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # seg_map2 = np.load('seg_map2.npy')
    # seg_map2 = np.expand_dims(seg_map2, axis=2)

    # white_bg2 = np.where((seg_map2==0), [255,255,255], img2)
    # remove_human2 = np.where((seg_map2==15), [255,0,0], img2)
    # remove_cat = np.where((seg_map2!=0) & (seg_map2!=15) , [255,0,0], img2)
    
    # #Plot the results
    # fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(10, 8))
    
    # axs[0,0].imshow(img)
    # axs[0,0].set_title('Image')
    # axs[0,0].axis('off')

    # axs[0,1].imshow(seg_map)
    # axs[0,1].set_title("Segmentation map")
    # axs[0,1].axis('off')

    # axs[0,2].imshow(img)
    # axs[0,2].imshow(seg_map, alpha=0.5)
    # axs[0,2].set_title("Overlaid")
    # axs[0,2].axis('off')

    # axs[1,0].imshow(white_bg)
    # axs[1,0].set_title("White background")
    # axs[1,0].axis('off')

    # axs[1,1].imshow(remove_human)
    # axs[1,1].set_title("Remove human")
    # axs[1,1].axis('off')

    # axs[1,2].imshow(remove_non_human)
    # axs[1,2].set_title("Remove non human")
    # axs[1,2].axis('off')

    # axs[2,0].imshow(img2)
    # axs[2,0].set_title("Bottles, human & background")
    # axs[2,0].axis('off')

    # axs[2,1].imshow(seg_map2)
    # axs[2,1].set_title("Segmentation map2")
    # axs[2,1].axis('off')

    # axs[2,2].imshow(img2)
    # axs[2,2].imshow(seg_map2, alpha=0.5)
    # axs[2,2].set_title("Overlaid")
    # axs[2,2].axis('off')

    # axs[3,0].imshow(white_bg2)
    # axs[3,0].set_title("White background")
    # axs[3,0].axis('off')

    # axs[3,1].imshow(remove_human2)
    # axs[3,1].set_title("Remove human")
    # axs[3,1].axis('off')

    # axs[3,2].imshow(remove_cat)
    # axs[3,2].set_title("Remove cat")
    # axs[3,2].axis('off')
    
    # plt.tight_layout()
    # plt.show()


    
    # for i, perc in enumerate([y/4 for y in range(3)], 2):
    #     print(i, perc)

    for style_image in Path("style_transfer/images/style_images").iterdir():
         print(style_image.name)

    # for style_image in style_images_folder.iterdir():
    #     print(style_image.name, (style_images_folder / style_image).as_posix())

    


    
