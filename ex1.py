import cv2
import numpy as np
"""
-Detect the stars in both images using a star detection algorithm such as the Hough transform .
-Extract the feature descriptors of each star detected in both images using a feature descriptor algorithm such as SIFT.
-Match the feature descriptors of the stars in both images using a matching algorithm such as FLANN.
-Filter the matched feature descriptors based on a certain threshold, such as the ratio test.
-Compute the transformation matrix between the two images using the matched feature descriptors 
 and a geometric transformation algorithm such as RANSAC or homography estimation.
"""
# Load the two images
img1 = cv2.imread("image1.jpg")
img2 = cv2.imread("image2.jpg")

# Convert images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Detect the keypoints using SIFT
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# Match the keypoints using FLANN
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Filter the matches using Lowe's ratio test
matchesArr = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        matchesArr.append(m)

# Estimate the homography matrix using RANSAC
if len(matchesArr) > 10:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matchesArr]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matchesArr]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
else:
    print("Not enough good matches to estimate homography")

# Convert the matched keypoints to (x, y) coordinates
matches_xy = []
for match in matchesArr:
    x1, y1 = kp1[match.queryIdx].pt
    x2, y2 = kp2[match.trainIdx].pt
    matches_xy.append(((x1, y1), (x2, y2)))

# Apply the homography matrix to the keypoints in the first image
if M is not None:
    warped_pts = cv2.perspectiveTransform(src_pts, M)
else:
    warped_pts = None

# Display the matches
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matchesArr, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow("Matches", img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()
