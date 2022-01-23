import cv2
import numpy as np

COCO_BODY_PARTS = ['nose', 'neck',
                   'right_shoulder', ' right_elbow', 'right_wrist',
                   'left_shoulder', 'left_elbow', 'left_wrist',
                   'right_hip', 'right_knee', 'right_ankle',
                   'left_hip', 'left_knee', 'left_ankle',
                   'right_eye', 'left_eye', 'right_ear', 'left_ear', 'background'
                   ]


def skeleton(image,coordinates, x_ratio, y_ratio):
	background=image

	def coordinate_scaler(coordinates):
		return tuple([int(coordinates[0] * x_ratio), int(coordinates[1] * y_ratio)])
	# i='person18'
	# print(coordinates[i])
	for i in coordinates.keys():
		print("COORDINATES", coordinates)
		if tuple(coordinates[i]['nose'])!=(-1,-1) and tuple(coordinates[i]['neck'])!=(-1,-1):
			cv2.line(background,coordinate_scaler(coordinates[i]['nose']),coordinate_scaler(coordinates[i]['neck']),color=(0,0,255),thickness=3)

		if tuple(coordinates[i]['nose'])!=(-1,-1) and tuple(coordinates[i]['right_eye'])!=(-1,-1):
			cv2.line(background,coordinate_scaler(coordinates[i]['nose']),coordinate_scaler(coordinates[i]['right_eye']),color=(255,255,255),thickness=3)

		if tuple(coordinates[i]['nose'])!=(-1,-1) and tuple(coordinates[i]['left_eye'])!=(-1,-1):
			cv2.line(background, coordinate_scaler(coordinates[i]['nose']),coordinate_scaler(coordinates[i]['left_eye']),color=(255,255,255),thickness=3)

		if tuple(coordinates[i]['right_eye'])!=(-1,-1) and tuple(coordinates[i]['right_ear'])!=(-1,-1):
			cv2.line(background,coordinate_scaler(coordinates[i]['right_eye']),coordinate_scaler(coordinates[i]['right_ear']),color=(255,255,255),thickness=3)

		if tuple(coordinates[i]['left_eye'])!=(-1,-1) and tuple(coordinates[i]['left_ear'])!=(-1,-1):
			cv2.line(background,coordinate_scaler(coordinates[i]['left_eye']),coordinate_scaler(coordinates[i]['left_ear']),color=(255,255,255),thickness=3)

		if tuple(coordinates[i]['neck'])!=(-1,-1) and tuple(coordinates[i]['right_shoulder'])!=(-1,-1):
			cv2.line(background,coordinate_scaler(coordinates[i]['neck']),coordinate_scaler(coordinates[i]['right_shoulder']),color=(255,255,255),thickness=3)

		if tuple(coordinates[i]['neck'])!=(-1,-1) and tuple(coordinates[i]['left_shoulder'])!=(-1,-1):
			cv2.line(background,coordinate_scaler(coordinates[i]['neck']),coordinate_scaler(coordinates[i]['left_shoulder']),color=(255,0,85),thickness=3)

		if tuple(coordinates[i]['right_shoulder'])!=(-1,-1) and tuple(coordinates[i]['right_elbow'])!=(-1,-1):
			cv2.line(background,coordinate_scaler(coordinates[i]['right_shoulder']),coordinate_scaler(coordinates[i]['right_elbow']),color=(255,170,0),thickness=3)

		if tuple(coordinates[i]['right_elbow'])!=(-1,-1) and tuple(coordinates[i]['right_wrist'])!=(-1,-1):
			cv2.line(background,coordinate_scaler(coordinates[i]['right_elbow']),coordinate_scaler(coordinates[i]['right_wrist']),color=(255,255,0),thickness=3)

		if tuple(coordinates[i]['left_shoulder'])!=(-1,-1) and tuple(coordinates[i]['left_elbow'])!=(-1,-1):
			cv2.line(background,coordinate_scaler(coordinates[i]['left_shoulder']),coordinate_scaler(coordinates[i]['left_elbow']),color=(255,255,255),thickness=3)

		if tuple(coordinates[i]['left_elbow'])!=(-1,-1) and tuple(coordinates[i]['left_wrist'])!=(-1,-1):
			cv2.line(background,coordinate_scaler(coordinates[i]['left_elbow']),coordinate_scaler(coordinates[i]['left_wrist']),color=(255,255,255),thickness=3)

		if tuple(coordinates[i]['neck'])!=(-1,-1) and tuple(coordinates[i]['right_hip'])!=(-1,-1):
			cv2.line(background,coordinate_scaler(coordinates[i]['neck']),coordinate_scaler(coordinates[i]['right_hip']),color=(0,255,0),thickness=3)

		if tuple(coordinates[i]['neck'])!=(-1,-1) and tuple(coordinates[i]['left_hip'])!=(-1,-1):
			cv2.line(background,coordinate_scaler(coordinates[i]['neck']),coordinate_scaler(coordinates[i]['left_hip']),color=(0,255,255),thickness=3)

		if tuple(coordinates[i]['right_hip'])!=(-1,-1) and tuple(coordinates[i]['right_knee'])!=(-1,-1):
			cv2.line(background,coordinate_scaler(coordinates[i]['right_hip']),coordinate_scaler(coordinates[i]['right_knee']),color=(0,102,255),thickness=3)

		if tuple(coordinates[i]['right_knee'])!=(-1,-1) and tuple(coordinates[i]['right_ankle'])!=(-1,-1):
			cv2.line(background,coordinate_scaler(coordinates[i]['right_knee']),coordinate_scaler(coordinates[i]['right_ankle']),color=(255,102,255),thickness=3)

		if tuple(coordinates[i]['left_hip'])!=(-1,-1) and tuple(coordinates[i]['left_knee'])!=(-1,-1):
			cv2.line(background,coordinate_scaler(coordinates[i]['left_hip']),coordinate_scaler(coordinates[i]['left_knee']),color=(255, 0, 170),thickness=3)

		if tuple(coordinates[i]['left_knee'])!=(-1,-1) and tuple(coordinates[i]['left_ankle'])!=(-1,-1):
			cv2.line(background,coordinate_scaler(coordinates[i]['left_knee']),coordinate_scaler(coordinates[i]['left_ankle']),color=(0,85,255),thickness=3)

		#cv2.putText(background, i,tuple(coordinates[i]['right_eye']),0, 5e-3 * 200, (0,0,255),2)


	return background