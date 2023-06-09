# Example usage:
# For image inference: python inference.py --source image --path "path to the image" --save False
# For camera inference: python inference.py --source 0

import numpy as np
from facenet_pytorch import MTCNN
import cv2
import time
import torch
import argparse
from emotions_model.model import EmotionsModel


def transform(im,device):
	im = cv2.resize(im, (48,48))
	im = torch.tensor(im/255.)
	im = im.permute(2,0,1)
	return im.float().to(device).unsqueeze(0)


def inference(frame,mtcnn,model,model_em,Genders,Moods):
	model.eval()
	model_em.eval()
	frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
	frame_BGR = frame.copy()
	boxes, probs = mtcnn.detect(frame, landmarks=False)
	preds = None
	
	if (probs.all() != None and probs.all() > 0.8):
		for x1,y1,x2,y2 in boxes:
			x1,x2,y1,y2 = int(x1),int(x2),int(y1),int(y2)
			face = frame_BGR[y1:y2,x1:x2]
			face_cp = face
			if face.shape[0] == 0 or face.shape[1] == 0:
				continue
			face = np.transpose(cv2.resize(face, (128,128)), (2,0,1))
			face = np.expand_dims(face, 0).astype(np.float32)/255.
			predictions = model(torch.tensor(face).to(device)).detach().cpu().numpy()
			gender = int(np.argmax(predictions[0, :2]))
			age = int(predictions[0, 2])
			emotion = torch.argmax(model_em(transform(face_cp,device))).cpu().numpy().item()
			preds = (['Male', 'Female'][gender], age, Moods[emotion])
			cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,150),1)
			cv2.putText(frame, 'Gender: {}, Age: {}, Mood: {}'.format(['Male', 'Female'][gender], age, Moods[emotion]), (x1,y1-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 150))
	frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
	return frame, preds


def photo_inference(device,mtcnn,model,model_em,Genders,Moods,opt,save,path):
	image = cv2.imread(path)
	
	image, _ = inference(image,mtcnn,model,model_em,Genders,Moods)
	if save:
		cv2.imwrite(r"C:\Users\fano\Desktop\im_new.png", image)
	while True:
		cv2.imshow('inference',image)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break


def camera_inference(device,mtcnn,model,model_em,Genders,Moods,opt,source=0):

	prev_frame_time = 0
	next_frame_time = 0

	cap = cv2.VideoCapture(source) 
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			break
		
		frame, predictions = inference(frame,mtcnn,model,model_em,Genders,Moods)

		next_frame_time = time.time()
		try:
			fps = str(round(1/(next_frame_time-prev_frame_time),2))
		except ZeroDivisionError:
			fps = ""
		prev_frame_time = next_frame_time

		cv2.putText(frame, fps, (7,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)
		cv2.imshow("Face Recognition", frame)

		key = cv2.waitKey(1)
		if key & 0xFF == ord('q'):
			break
		elif key & 0xFF == ord('x'):
			print(predictions)

	cap.release()
	cv2.destroyAllWindows()


def parse_opt():
	parser = argparse.ArgumentParser()
	parser.add_argument('--source', '-s', type=str, default='camera',help='camera|image|video')
	parser.add_argument('--path', '-p', type=str, default=None, help='path to the image')
	parser.add_argument('--save', '-sv', type=bool, default=False, help='save image - True/False')
	opt = parser.parse_args()
	return opt


if __name__ == '__main__':

	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	print("Initializing models ...")
	mtcnn = MTCNN(image_size=160, margin=14, min_face_size=40,device=device, post_process=False)
	print("Loading weights ...")
	model = torch.load('weights/best.pt',map_location=torch.device(device)).to(device)
	model_em = EmotionsModel().to(device)
	model_em.load_state_dict(torch.load('weights/model66.pt',map_location=torch.device(device)))

	Genders = {0: 'male', 1: 'female'}
	Moods = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'neutral', 6: 'surprise'}

	opt = parse_opt()
	if vars(opt)['source'] == 'camera':
		print("Starting camera ...")
		print('Press "Q" to quit')
		camera_inference(device,mtcnn,model,model_em,Genders,Moods,opt,0)
	elif vars(opt)['source'] == 'video' and vars(opt)['path'] != None:
		print("Predictions on video")
		camera_inference(device,mtcnn,model,model_em,Genders,Moods,opt,vars(opt)['path'])
	else:
		print("Making prediction on photo ...")
		photo_inference(device,mtcnn,model,model_em,Genders,Moods,opt,vars(opt)['save'],vars(opt)['path'])
	print("Done!")
