import cv2
import numpy as np
"""
part2:
1.Read in the image of stars and convert it into grayscale.
2.Detect the stars in the image using a star detection algorithm such as the Hough transform or a blob detector.
3.For each star detected, extract its centroid coordinates (x,y) and radius (r) using a shape detection algorithm such as the circular Hough transform or contour detection.
4.Compute the brightness (b) of each star by averaging the pixel values within its radius.
5.Save the coordinates (x,y,r,b) of each star into a file.
"""
# Load the image
image = cv2.imread("stars.jpg")

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect the stars in the image using the Hough transform
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=50, param2=30, minRadius=0, maxRadius=0)

# Save the coordinates (x,y,r,b) of each star into a file
with open("stars_coordinates.txt", "w") as f:
    for circle in circles[0]:
        x, y, r = circle.astype(int)
        brightness = np.mean(gray[y-r:y+r, x-r:x+r])
        f.write(f"{x},{y},{r},{brightness}\n")
