import cv2
from matplotlib import pyplot as plt
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
sam = sam_model_registry["default"](checkpoint="sam_vit_h_4b8939.pth")
mask_generator = SamAutomaticMaskGenerator(sam)

image = cv2.imread('test.png')

masks = mask_generator.generate(image)

print(len(masks))

plt.figure(figsize=(20,20))
plt.imshow(image)

plt.axis('off')
plt.show() 
