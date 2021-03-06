# USAGE
# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav

# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2
import time

def sound_alarm(path):
	# play an alarm sound
	playsound.playsound(path)

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear
 
# 인자값, default로 설정 완료
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", default="shape_predictor_68_face_landmarks.dat",
	help="path to facial landmark predictor")
ap.add_argument("-a", "--alarm", default="alarm.WAV",
	help="path alarm .WAV file")
ap.add_argument("-w", "--webcam", type=int, default=0,
	help="index of webcam on system")
args = vars(ap.parse_args())
 
##눈의 감긴 정도 구별 변수
EYE_AR_THRESH = 0.22


##연속적으로 눈을 감은 횟수
EYE_AR_CONSEC_FRAMES = 48
consecutive_eyes_closed = 1

##프레임 반복 횟수(while문 반복 횟수)
# frame_number=0
## 평균 눈 깜빡임 횟수
Blink_avg_number = 0

#x초 마다 깜빡임 수를 구하기
x_second = 10
x_repeat = 0

#전체 눈 깜빡임 횟수
TOTAL = 0
##깜빡인 시간차를 위한 변수
Eye_blinking_time = 0

##눈 인식 되지 않는 시간##
Eye_Notrecognition_time = 0

COUNTER = 0
ALARM_ON = False

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)
start_time=0
# end_time=0

# start = time.time()
# loop over frames from the video stream
while True:
	now = time.time()
	print("while문 실행 시간"+str(now-start))
	if (now-start)>=float(x_second) and ((now-start)%x_second) < float(1):
		x_repeat+=1
		print("******20초**********")
		Blink_avg_number = TOTAL/x_repeat
		print("총 깜빡인 수:"+str(TOTAL))
		print("20초 반복 수:"+str(x_repeat))
		print("평균 깜빡인 수 : "+str(Blink_avg_number))
		start=time.time()
	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	##눈 인식이 5초 이상 되지 않을 경우 알림 울리기
	if not rects:
		Eye_Notrecognition_time += 1
		print(Eye_Notrecognition_time)
	if Eye_Notrecognition_time >=10:
		if not ALARM_ON:
			ALARM_ON = True
			# check to see if an alarm file was supplied,
			# and if so, start a thread to have the alarm
			# sound played in the background
			if args["alarm"] != "":
				t = Thread(target=sound_alarm,
						   args=(args["alarm"],))
				t.deamon = True
				t.start()
		cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

	# loop over the face detections
	for rect in rects:
		Eye_Notrecognition_time=0
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0

		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
		if ear < EYE_AR_THRESH:
			COUNTER += 1
			print("counter 수 : "+str(COUNTER))

			#################### blink_time############################
			# if COUNTER == 1:
				# end_time = time.time()
				# blink_interval_time=(end_time-start_time)
				# print("계산한거 : "+str(blink_interval_time))
			###########################################################

			# if the eyes were closed for a sufficient number of
			# then sound the alarm
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				# if the alarm is not on, turn it on
				if not ALARM_ON:
					ALARM_ON = True

					# check to see if an alarm file was supplied,
					# and if so, start a thread to have the alarm
					# sound played in the background
					if args["alarm"] != "":
						t = Thread(target=sound_alarm,
							args=(args["alarm"],))
						t.deamon = True
						t.start()

				# draw an alarm on the frame
				cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		# otherwise, the eye aspect ratio is not below the blink
		# threshold, so reset the counter and alarm
		else:
			if COUNTER>=consecutive_eyes_closed:
				TOTAL+=1
			start_time = time.time()
			COUNTER = 0
			ALARM_ON = False

		# draw the computed eye aspect ratio on the frame to help
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

		#눈 깜빡인 횟수 화면 출력
		cv2.putText(frame, "Blinks: {}".format(TOTAL), (300, 50),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key&0xFF == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()