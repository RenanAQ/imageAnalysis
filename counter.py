import cv2
import numpy as np
import matplotlib.pyplot as plt


imagesArray = ['shelfCans-2.jpg', 'shelfCans-4.jpg', 'shelfCans-6.jpg', 'shelfCans4-2.jpg']
imagesArray2 = ['./jpegmini_optimized-2/IMG_4974.jpg', './jpegmini_optimized-2/IMG_4975.jpg']

# Loop over a range of min_area values
for j in range(0, 2000, 100):
    i = 0

    for imageName in imagesArray2:

        image = cv2.imread(imageName)

        if image is None:
            print(f"Error: Image '{imageName}' not found or unable to open.")
            continue  

        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)

        
        _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)

        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        
        edges = cv2.Canny(morphed, 50, 150)

        
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        
        min_area = j  
        max_area = 10000  
        filtered_contours = [c for c in contours if min_area < cv2.contourArea(c) < max_area]

        
        number_of_cans = len(filtered_contours)

        # Display the result
        if i < 1:
            print(f"Parameters min_area {min_area}:")
            print(f"Parameters max_area {max_area}:")
        i += 1    
        print(f"Number of cans detected from {imageName}: {number_of_cans}")

        
        output_image = image.copy()
        cv2.drawContours(output_image, filtered_contours, -1, (0, 255, 0), 2)

        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        plt.subplot(1, 2, 2)
        plt.title("Detected Cans")
        plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))

        
        plt.show()