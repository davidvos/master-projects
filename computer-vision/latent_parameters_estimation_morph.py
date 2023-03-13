from supplemental_code import *
import cv2
import dlib
import openface

def visualise_single_picture():
	img_path = "marcel_image.jfif"
	#img_path = "1_oog_man.jpg"
	img = cv2.imread(img_path)

	# Face Detector
	detector = dlib.get_frontal_face_detector()
	# Landmark Detector


	def predict_landmarks(img):
		dets = detector(img, 1)
		if len(dets) < 1:
			return None # Face Not Found
		  
		print("Found %d faces" % len(dets))
		d = dets[0]
		gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		landmarks = predictor.findLandmarks(gray_img, d)
		return np.asarray(landmarks)


	predictor = openface.AlignDlib("shape_predictor_68_face_landmarks.dat")
	marcel_landmarks = predict_landmarks(img)

	import matplotlib.pyplot as plt
	#plt.imshow(img[...,::-1])
	#plt.show()

	def visualize_landmarks(img, landmarks, radius=2):
	  new_img = np.copy(img)
	  h, w, _ = new_img.shape
	  for x, y in landmarks:
	    x = int(x)
	    y = int(y)
	    new_img[max(0,y-radius):min(h-1,y+radius),max(0,x-radius):min(w-1,x+radius)] = (255, 0, 0)
	  plt.imshow(new_img[...,::-1])
	  plt.show()

	visualize_landmarks(img, marcel_landmarks)

visualise_single_picture()