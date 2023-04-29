import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import os
import cv2
from tqdm import tqdm, tnrange

import torchvision
from torchvision import transforms, models, datasets

import matplotlib.pyplot as plt
from PIL import Image
from torch import optim # not needed?

from glob import glob
import pandas as pd

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler


from dataset import Emotions
from model import EmotionsModel


device = 'cuda' if torch.cuda.is_available() else 'cpu'
folder = r'data/train/'


emotions_dataset = Emotions(folder)
validation_split = 0.2
seed = 12

indxs = list(range(len(emotions_dataset)))
split = int(np.floor(validation_split*len(emotions_dataset)))
np.random.seed(seed)
np.random.shuffle(indxs)
trn_ind, val_ind = indxs[split:], indxs[:split]

trn_s = SubsetRandomSampler(trn_ind)
val_s = SubsetRandomSampler(val_ind)

trn_dl = DataLoader(emotions_dataset, batch_size=64, sampler=trn_s)
vld_dl = DataLoader(emotions_dataset, batch_size=16, sampler=val_s)




model = EmotionsModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss()


torch.save(model.state_dict(), 'model.pt')
def train_batch(x, y, model, loss, optim):
	model.train(),
	prediction = model(x)
	y = torch.squeeze(y)
	loss_value = loss(prediction, y)
	loss_value.backward()
	optim.step()
	optim.zero_grad()
	return loss_value.item()


def accuracy(x, y, model):
	model.eval()
	prediction = model(x)
	is_correct = torch.argmax(prediction) == y
	return is_correct.cpu().numpy().tolist()


train_accuracies, train_losses = [], []
val_accuracies = []

for epoch in range(100):
	print(f" epoch {epoch + 1}/100")
	train_epoch_losses, train_epoch_accuracies = [], []
	val_epoch_accuracies = []
	for ix, batch in enumerate(iter(trn_dl)):
		x, y = batch
		batch_loss = train_batch(x, y, model, criterion, optimizer)
		train_epoch_losses.append(batch_loss)
	for ix, batch in enumerate(iter(trn_dl)):
		x, y = batch
		is_correct = accuracy(x, y, model)
		train_epoch_accuracies.extend(is_correct)

	train_epoch_loss = np.array(train_epoch_losses).mean()
	train_epoch_accuracy = np.mean(train_epoch_accuracies)
	for ix, batch in enumerate(iter(vld_dl)):
		x, y = batch
		val_acc = accuracy(x, y, model)
		val_epoch_accuracies.extend(val_acc)
	val_epoch_accuracy = np.mean(val_epoch_accuracies)
	train_losses.append(train_epoch_loss)
	train_accuracies.append(train_epoch_accuracy)
	val_accuracies.append(val_epoch_accuracy)
	if epoch % 5 == 1:
		torch.save(model.state_dict(), f'model{epoch}.pt')
	print("loss at {} epoch is ".format(epoch + 1) + str(train_epoch_loss))


torch.save(model.state_dict(), 'model_last.pt')