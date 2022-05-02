import os
from PIL import Image
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import tqdm
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler

class FER_Dataset(Dataset):
	def __init__(self, img_data, img_path, transform=None):
		self.img_path = img_path
		self.transform = transform
		self.img_data = img_data
		
	def __len__(self):
		return len(self.img_data)
	
	def __getitem__(self, index):
		img_name = os.path.join(self.img_path, self.img_data.loc[index, 'Labels'],
												self.img_data.loc[index, 'Images'])
		
		image = Image.open(img_name)
		label = torch.tensor(self.img_data.loc[index, 'encoded_labels'])
		
		image = self.transform(image)
		
		return image, label



def make_dataloader(shuffle_dataset=True, validation_split=.3, batch_size=32, transform):
	# make dataset list
	images = []
	labels = []
	emotions = os.listdir(data_path)
	for emotion in emotions:
		filenames = os.listdir(data_path + emotion + '/')
		for filename in filenames:
			images.append(filename)
			labels.append(emotion)

	emotion_data = {'Images':images, 'Labels':labels}
	emotion_data = pd.DataFrame(emotion_data)

	lb = LabelEncoder()
	emotion_data['encoded_labels'] = lb.fit_transform(emotion_data['Labels'])

	random_seed = 42

	# make sampler
	dataset_size = len(emotion_data)
	indices = list(range(dataset_size))
	split = int(np.floor(validation_split * dataset_size))

	if shuffle_dataset:
		np.random.seed(random_seed)
		np.random.shuffle(indices)

	train_indices, val_indices = indices[split:], indices[:split]

	train_sampler = SubsetRandomSampler(train_indices)
	valid_sampler = SubsetRandomSampler(val_indices)
	
# transform
transform = transforms.Compose([transforms.Resize((224,224)),
transforms.ToTensor(),
transforms.Normalize((0.5, ), (0.5, ))])

	
def make_loader(data, data_path, transform, batch_size=32):
	
	dataset = FER_Dataset(data, data_path, transform)
	
	train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler
