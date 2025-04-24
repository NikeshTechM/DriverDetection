import cv2
import mediapipe as mp
import time
import argparse
import pickle
import face_recognition
import os
from   flask import Flask, render_template, Response, jsonify
import warnings
import socket
import threading
from collections import deque

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--port', default = 5000,  required = False)
parser.add_argument('-t', '--timer',default = 10 ,required= False)
args = vars(parser.parse_args())
port = int(args['port'])
initial_timer_value = int(args['timer'])

app = Flask(__name__)

font_scale = 0.5
position1  = (50,50)
position2  = (50,70)
font       = cv2.FONT_HERSHEY_SIMPLEX
color      = (0,255,0)
thickness  = 1
linetype   = cv2.LINE_AA
start_time = time.time()
fps        = 0
count      = 0
bb_face    = None
cam_width  = 0
cam_height = 0
user       = 'Driver'
name_list  = deque(maxlen = 10)
for _ in range(10):
	name_list.append('Driver')

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh()
face_detection = mp.solutions.face_detection.FaceDetection(model_selection = 0, min_detection_confidence = 0.85)

file    = open(os.path.join(os.getcwd(), "trained_knn_model.clf"), 'rb')
knn_clf = pickle.load(file)
host = socket.gethostbyname(socket.gethostname())
print('host ip', host)

def transform(image):
	global count
	global font_scale
	global position1 
	global position2 
	global font      
	global color     
	global thickness 
	global linetype  
	global start_time
	global fps
	global bb_face
	global cam_width
	global cam_height
	global face_detection
	global knn_clf
	global user
	global name_list

	count += 1

	image = cv2.flip(image, 1)
	# Convert the image to RGB
	image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# FPS calculation
	if count % 5 == 0:
		elapsed_time = time.time()-start_time
		fps = int(5 / elapsed_time)
		start_time = time.time()
	cv2.putText(image, f'FPS:{fps:.2f}', position1, font, font_scale, color, thickness, linetype)

	if count % 15 == 0:
		count = 1
		# Face Detection using MediaPipe Face-Detection
		results = face_detection.process(image_rgb)
		if results.detections:
			if len(results.detections) == 1: # In frame there should be only one person
				x = int(results.detections[0].location_data.relative_bounding_box.xmin   * cam_width)
				w = int(results.detections[0].location_data.relative_bounding_box.width  * cam_width)
				y = int(results.detections[0].location_data.relative_bounding_box.ymin   * cam_height)
				h = int(results.detections[0].location_data.relative_bounding_box.height * cam_height)

				print('Mediapipe locations: ', y, x+w, y+h, x)
				X_face_locations = [(y, x+w, y+h, x)]

				faces_encodings = face_recognition.face_encodings(image_rgb, known_face_locations=X_face_locations)

				# Use the KNN model to find the best matches for the test face
				closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
				are_matches = [closest_distances[0][i][0] <= 0.45 for i in range(len(X_face_locations))]
			
				predictions = [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

				for name, (top, right, bottom, left) in predictions:
					print(name)
					name_list.append('Guest' if name == 'unknown' else name)
					if len(set(list(name_list)[:3])) == 1:
						user = name_list[1]

	# Draw face mesh using MediaPipe-FaceMesh
	results = face_mesh.process(image_rgb)
	if results.multi_face_landmarks:
		for face_landmarks in results.multi_face_landmarks:
			for landmark in face_landmarks.landmark:
				# Extracting landmark information
				x = int(landmark.x * image.shape[1])
				y = int(landmark.y * image.shape[0])
				cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
			
	return image

def generate_frames():
	global cam_height
	global cam_width
	global user

	def socket_server():
		server_socket  = socket.socket()  # get instance
		try:
			# get the hostname
			host = socket.gethostbyname(socket.gethostname()) #'10.42.0.1'
			port = 42000  # initiate port no above 1024

			# look closely. The bind() function takes tuple as argument
			server_socket.bind((host, port))  # bind host address and port together

			# configure how many client the server can listen simultaneously
			server_socket.listen(1)

			while True:
				conn, address = server_socket.accept()  # accept new connection
				print("Connection started from: " + str(address))

				while True:
					# receive data stream. It won't accept data packet greater than 1024 bytes
					data = conn.recv(1024).decode()
					if not data:
						break # if data is not received break
					if 'spit' in data.lower():
						print("from connected user: " + str(data))
						conn.send(f'{user.capitalize()}\n'.encode())
				conn.close()  # close the connection
				print("Connection closed from: " + str(address))
		except Exception as e:
			print(f'Error in socket connection: {e}')
			exit(1)

	threading.Thread(target = socket_server).start()

	camera = cv2.VideoCapture(0)
	cam_width  = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
	cam_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`

	while True:
		success, frame = camera.read()
		
		if not success: break

		frame = transform(frame)
		ret, buffer = cv2.imencode('.png', frame)
		frame = buffer.tobytes()
		yield (b'--frame\r\n'
		 	   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

	camera.release()

@app.route('/')
def index():
	return render_template('index.html',timer_value=initial_timer_value)

@app.route('/video_feed')
def video_feed():
	return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_recognized_face', methods=['GET'])
def get_recognized_face():
	global user
	print('I am in the get_recognized_face')
	final_str = ''
	if user == 'Guest':
		final_str = 'Unidentified User Detected'
	else:
		final_str = 'Welcome ' + user.capitalize()
	return jsonify({'faceName': final_str})

if __name__ == "__main__":
	# Always use a port greater than 1024.
	app.run('0.0.0.0', port, True)
