'''
Face Detection using Azure Cognitive Services
Demonstration link - https://youtu.be/e0xqMmpjYL8

Environment Information:
-----------------------
	-- Python 3.6.9
	-- azure-cognitiveservices-vision-face==0.5.0
	-- cv2==4.1.1
	-- msrest==0.6.21
	-- threading (for multi-threading)

'''

import cv2
import time
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials

import threading

subscription_key="YOUR_SUBSCRIPTION_KEY"
endpoint="YOUR_ENDPOINT"
face_client=FaceClient(endpoint, CognitiveServicesCredentials(subscription_key))

class FaceDetector(threading.Thread):
	'''
	FaceDetector class inherits the Thread class which enables it to 
	run multiple functions simultaneously.

	We will run one function to render real-time video and
	the other function to call face detector API and 
	render the results after every 1 second.

	'''
	def __init__(self, threadID, name):
		'''
		Arguments
		---------
			threadID (int) : unique thread id
			name (str) : title of the webcam

		Returns
		-------
			None.
		'''
		threading.Thread.__init__(self)
		self.threadID = threadID
		self.name = name

		# Position of the sub-headings
		self.org = (50, 30)
		# Font style of the text
		self.font = cv2.FONT_HERSHEY_SIMPLEX
		# Font size
		self.font_scale = 1
		# Font color
		self.color = (0, 255, 0)
		# Bounding box thickness
		self.thickness = 2
		# age
		self.age = ""
		# gender
		self.gender = ""

		# Initialize frames to None
		self.frame2 = None
		self.frame = None

		# Initialze and launch web camera
		self.cap = cv2.VideoCapture()
		self.cap.open(0)

		# Emotion counter
		self.counter = 0

	# Captures and displays real-time video
	def run(self):
		# Initialize frames to legitimate values 
		_, self.frame = self.cap.read()
		self.frame2 = self.frame.copy()

		# Forever loop
		while (True):
			# Capture the frame
			_, frame = self.cap.read()
			# Copy the frame
			self.frame = frame.copy()
			# Put heading on the frame 
			frame = cv2.putText(frame, "Real Time", self.org, self.font, self.font_scale, self.color, self.thickness, cv2.LINE_AA)
			# Prepare split display
			frame_full = cv2.hconcat([frame, self.frame2])
			# Display
			cv2.imshow(self.name, frame_full)
			# Add a 1 ms delay
			# Else the video won't render
			cv2.waitKey(1)
			# Check if the window is closed
			# If yes, break.
			if cv2.getWindowProperty(self.name, cv2.WND_PROP_VISIBLE) <1:
				break
		# Destroy all the windows
		cv2.destroyAllWindows()

	# Detects face and attributes
	def detect_faces(self, local_image):
		face_attributes = ['emotion', 'age', 'gender']
		# Call the face detection web service
		detected_faces = face_client.face.detect_with_stream(local_image, return_face_attributes=face_attributes, detection_model='detection_01')
		return detected_faces

	# The detector function
	def detector(self):
		emotions_ref = ["neutral", "sadness", "happiness", "disgust", "contempt", "anger", "surprise", "fear"]
		emotions_found  = []
		# Forever loop
		while (True):
			# Add a delay of 1 second
			time.sleep(1)
			# Copy the frame at the current moment
			frame = self.frame.copy()
			# Save it as an image
			cv2.imwrite('test.jpg', frame)
			# Open the image
			local_image = open('test.jpg', "rb")
			# Pass the image to the detector webservice
			faces = self.detect_faces(local_image)
			# Check if the faces are found
			if (len(faces)>0):
				# Extract face attributes of the first face found
				age = faces[0].face_attributes.age
				gender = faces[0].face_attributes.gender
				gender = (gender.split('.'))[0]
				# Extrac the emotion out of the face attributes
				emotion = self.get_emotion(faces[0].face_attributes.emotion)
				# If the detected emotion is new, update the counter
				if(emotion[0] in emotions_ref):
					self.counter += 1
					emotions_ref.remove(emotion[0])
				# Extract the bounding box
				left, top, width, height = self.getRectangle(faces[0])
				# Draw the bounding box and
				# Display the results on screen
				frame = cv2.rectangle(frame, (left, top), (left+width, top+height+100), (255, 0, 0), 3)
				frame = cv2.rectangle(frame, (left, top+height), (left+width, top+height+100), (255, 0, 0), cv2.FILLED)
				frame = cv2.putText(frame, "age: "+str(int(age)), (left, top+height+20), self.font, 0.6, self.color, self.thickness, cv2.LINE_AA)
				frame = cv2.putText(frame, "gender: "+str(gender), (left, top+height+40), self.font, 0.6, self.color, self.thickness, cv2.LINE_AA)
				frame = cv2.putText(frame, "emotion: ", (left, top+height+60), self.font, 0.6, self.color, self.thickness, cv2.LINE_AA)
				frame = cv2.putText(frame, str(emotion[0]), (left, top+height+80), self.font, 0.6, self.color, self.thickness, cv2.LINE_AA)
				frame = cv2.putText(frame, "Face Detection", self.org, self.font, self.font_scale, self.color, self.thickness, cv2.LINE_AA)
				frame = cv2.putText(frame, "#emotions : "+str(self.counter), (400, 30), self.font,  self.font_scale, self.color, self.thickness, cv2.LINE_AA)
				self.frame2 = frame

	# Emotion extractor function
	def get_emotion(self, emotion_obj):
	    emotion_dict = dict()
	    emotion_dict['anger'] = emotion_obj.anger
	    emotion_dict['contempt'] = emotion_obj.contempt
	    emotion_dict['disgust'] = emotion_obj.disgust
	    emotion_dict['fear'] = emotion_obj.fear
	    emotion_dict['happiness'] = emotion_obj.happiness
	    emotion_dict['neutral'] = emotion_obj.neutral
	    emotion_dict['sadness'] = emotion_obj.sadness
	    emotion_dict['surprise'] = emotion_obj.surprise
	    emotion_name = max(emotion_dict, key=emotion_dict.get)
	    emotion_confidence = emotion_dict[emotion_name]
	    return emotion_name, emotion_confidence

	# Bounding box extractor function
	def getRectangle(self, faceDictionary):
	    rect = faceDictionary.face_rectangle
	    left = rect.left
	    top = rect.top
	    width = rect.width
	    height = rect.height
	    return left, top, width, height

# Create FaceDetector object
detector = FaceDetector(1, "Face Detection - Azure")
# Start the real-time capture and render
detector.start()
# Wait for half a second
time.sleep(0.5)
# Start the face detector function
detector.detector()