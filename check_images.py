import cv2
import os
import numpy as np

data_dir = './data1'
images = []

for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)
    if not os.path.isdir(class_path):
        continue
    
    for img_name in os.listdir(class_path)[:3]:  # Check first 3 images
        img_path = os.path.join(class_path, img_name)
        if os.path.isfile(img_path):
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                images.append(image)
                print(f"Class: {class_name}, Image: {img_name}, Shape: {image.shape}")

print(f"\nTotal images checked: {len(images)}")
if images:
    print(f"Sample shapes: {[img.shape for img in images[:5]]}")
