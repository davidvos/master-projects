import torch
import numpy as np 
from os import listdir
from os.path import isfile, join
import pickle

# to do: make it functional for batches (?) for now this should/could suffice (per 1 datapoint)
class Dataset(torch.utils.data.Dataset):
	def __init__(self, train_dir):
		self.train_dir = train_dir
		self.all_files =  [self.train_dir + "/" + f for f in listdir(self.train_dir) if isfile(join(self.train_dir, f))]

	def __len__(self):
		return len(self.all_files)

	def __getitem__(self, index):

		with open(self.all_files[index], 'rb') as load_file:
			data = pickle.load(load_file)

		x_tensor = torch.empty((1,512,768), dtype = torch.float64)
		y_tensor = torch.empty((1,6), dtype = torch.float64)
		for index, (id_, pair) in enumerate(data.items()):
			
			x_tensor = torch.cat((x_tensor,torch.tensor(pair['question'], dtype = torch.float64)), dim = 0)
			y_tensor = torch.cat((y_tensor, torch.tensor(pair['label'], dtype = torch.float64).unsqueeze(dim=0)), dim = 0)
		return x_tensor[1:], y_tensor[1:]