from utils import *
from matplotlib import pyplot as plt
import os
from playsound import playsound

import subprocess
from gtts import gTTS


max_val = 8
max_pt = -1
max_kp = 0

orb = cv2.ORB_create()
# orb is an alternative to SIFT

test_img = read_img('files/test_100_2.jpg')
#test_img = read_img('files/test_50_2.jpg')
#test_img = read_img('files/test_20_2.jpg')
#test_img = read_img('files/test_100_3.jpg')
#test_img = read_img('files/test_20_4.jpg')

# resizing must be dynamic
original = resize_img(test_img, 0.4)
display('original', original)

# keypoints and descriptors
# (kp1, des1) = orb.detectAndCompute(test_img, None)
(kp1, des1) = orb.detectAndCompute(test_img, None)

training_set = ['files/20.jpg', 'files/50.jpg', 'files/100.jpg', 'files/500.jpg']

for i in range(0, len(training_set)):
	# train image
	train_img = cv2.imread(training_set[i])

	(kp2, des2) = orb.detectAndCompute(train_img, None)

	# brute force matcher
	bf = cv2.BFMatcher()
	all_matches = bf.knnMatch(des1, des2, k=2)

	good = []
	# give an arbitrary number -> 0.789
	# if good -> append to list of good matches
	for (m, n) in all_matches:
		if m.distance < 0.789 * n.distance:
			good.append([m])

	if len(good) > max_val:
		max_val = len(good)
		max_pt = i
		max_kp = kp2

	print(i, ' ', training_set[i], ' ', len(good))

if max_val != 8:
	print(training_set[max_pt])
	print('good matches ', max_val)

	train_img = cv2.imread(training_set[max_pt])
	img3 = cv2.drawMatchesKnn(test_img, kp1, train_img, max_kp, good, 4)
	
	note = os.path.basename(training_set[max_pt]).split('.')[0]
	detected_value = note 
	print('\nDetected denomination: Rs. ', note)

	audio_file = f'audio/{detected_value}.mp3'
    
    # Generate the speech output
	speech_out = f"Detected currency value is Rs. {detected_value}"  # Use detected_value directly
	tts = gTTS(text=speech_out, lang="en")
	tts.save(audio_file)

    # Play the audio file
	playsound(audio_file)
	(plt.imshow(img3), plt.show())

else:
	print('No Matches')
