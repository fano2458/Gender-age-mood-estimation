import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

from glob import glob
import os
import cv2
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_data_dir = r'data/train'
test_data_dir = r'data/test'

folder = r'data/train/'

# 0 - angry, 1 - disgust, 2 - fear, 3 - happy, 4 - neutral, 5 - sad, 6 - surprise

class Emotions(Dataset):
	def __init__(self, folder):
		angry = [os.path.normpath(i) for i in glob(folder+'angry/*.jpg')]
		disgust = [os.path.normpath(i) for i in glob(folder+'disgust/*.jpg')]
		fear = [os.path.normpath(i) for i in glob(folder+'fear/*.jpg')]
		happy = [os.path.normpath(i) for i in glob(folder+'happy/*.jpg')]
		neutral = [os.path.normpath(i) for i in glob(folder+'neutral/*.jpg')]
		sad = [os.path.normpath(i) for i in glob(folder+'sad/*.jpg')]
		surprise = [os.path.normpath(i) for i in glob(folder+'sad/*.jpg')]
		self.paths = angry + disgust + fear + happy + neutral + sad + surprise
		self.targets = []
		print(self.paths[0])
		for file in self.paths:
			tmp = [0,0,0,0,0,0,0] 
			if file.startswith('data\\test\\angry') or file.startswith('data\\train\\angry'):
				tmp[0] = 1
			elif file.startswith('data\\test\\disgust') or file.startswith('data\\train\\disgust'):
				tmp[1] = 1
			elif file.startswith('data\\test\\fear') or file.startswith('data\\train\\fear'):
				tmp[2] = 1
			elif file.startswith('data\\test\\happy') or file.startswith('data\\train\\happy'):
				tmp[3] = 1
			elif file.startswith('data\\test\\neutral') or file.startswith('data\\train\\neutral'):
				tmp[4] = 1
			elif file.startswith('data\\test\\sad') or file.startswith('data\\train\\sad'):
				tmp[5] = 1
			elif file.startswith('data\\test\\surprise') or file.startswith('data\\train\\surprise'):
				tmp[6] = 1
			self.targets.append(tmp)

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, ix):
		file = self.paths[ix]
		target = self.targets[ix]
		im = cv2.imread(file)[:,:,::-1]
		im = cv2.resize(im, (48,48))
		im = torch.tensor(im/255.)
		im = im.permute(2,0,1)
		return im.float().to(device), torch.tensor([target]).float().to(device)
