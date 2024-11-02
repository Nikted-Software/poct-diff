import numpy as np
import cv2

image = cv2.imread("7921_crop.jpeg")  

# Load the segmentation data
data = np.load("7921_crop_seg.npy", allow_pickle=True).item()
print(data)  # Check the structure of data

binary_mask = data.get("masks")  
if binary_mask.shape != image.shape[:2]:
    binary_mask = cv2.resize(binary_mask, (image.shape[1], image.shape[0]))
binary_mask = (binary_mask > 0).astype(np.uint8) * 255
cv2.imshow("Binary Mask", binary_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

binary_mask = data.get("outlines")  
if binary_mask.shape != image.shape[:2]:
    binary_mask = cv2.resize(binary_mask, (image.shape[1], image.shape[0]))
binary_mask = (binary_mask > 0).astype(np.uint8) * 255
cv2.imshow("Binary Mask", binary_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()



