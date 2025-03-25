import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from playsound import playsound
from gtts import gTTS

# Function to load images from subdirectories
def load_training_images(base_path):
    training_set = []
    for subdir, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):  # Check for image files
                training_set.append(os.path.join(subdir, file))
    return training_set

# Function to apply a four-point perspective transformation
def four_point_transform(image, pts):
    rect = np.zeros((4, 2), dtype="float32")

    # Top-left point has the smallest sum (y + x)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    
    # Bottom-right point has the largest sum
    rect[2] = pts[np.argmax(s)]

    # Top-right point has the smallest difference (y - x)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    
    # Bottom-left point has the largest difference
    rect[3] = pts[np.argmax(diff)]

    # Unpack the rectangle points
    (tl, tr, br, bl) = rect

    # Compute the width and height of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Set the destination points for the transformation
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(rect, dst)

    # Apply the perspective transformation
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

# Path to your training dataset
base_path = 'C:/Users/Hariharane/Downloads/Train/'
training_set = load_training_images(base_path)

# Create an ORB detector
orb = cv2.ORB_create()

# Capture video from webcam
cap = cv2.VideoCapture(0)  # 0 for the default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame for better processing
    original = cv2.resize(frame, (640, 480))  # Adjust size as needed
    gray_frame = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    # Edge detection
    edged = cv2.Canny(blurred, 75, 200)

    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    screenCnt = None
    if contours:
        # Sort contours by area and keep the largest one
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        # Loop through contours to find a rectangle
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)  # Adjusted for better approximation
            if len(approx) == 4:
                screenCnt = approx
                break

        # If a rectangle was found, apply four-point transformation
        if screenCnt is not None:
            warped = four_point_transform(original, screenCnt.reshape(4, 2))

            # Keypoints and descriptors for the warped image
            kp1, des1 = orb.detectAndCompute(warped, None)

            max_val = 0
            max_pt = -1

            # Iterate through the training set to find matches
            for i, train_img_path in enumerate(training_set):
                train_img = cv2.imread(train_img_path)
                kp2, des2 = orb.detectAndCompute(train_img, None)

                # Brute force matcher
                bf = cv2.BFMatcher()
                all_matches = bf.knnMatch(des1, des2, k=2)

                good = []
                for (m, n) in all_matches:
                    if m.distance < 0.75 * n.distance:  # Adjusted ratio for better matching
                        good.append([m])

                if len(good) > max_val:
                    max_val = len(good)
                    max_pt = i

            if max_val > 10:  # Adjust this threshold as needed
                detected_image = training_set[max_pt]
                note = os.path.basename(os.path.dirname(detected_image))  # Get the subdirectory name
                print(f'\nDetected denomination: Rs. {note}')

                # Generate and play audio output
                audio_file = f'audio/{note}.mp3'
                speech_out = f"Detected currency value is Rs. {note}"
                tts = gTTS(text=speech_out, lang="en")
                tts.save(audio_file)
                playsound(audio_file)

                # Draw matches and display
                img3 = cv2.drawMatchesKnn(warped, kp1, cv2.imread(detected_image), kp2, good, None, flags=2)
                plt.imshow(img3)
                plt.title(f'Detected: Rs. {note}')
                plt.show()
                break  # Exit after successful detection

    # Display the original frame
    cv2.imshow("Webcam Feed", original)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
