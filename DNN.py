import torch.nn as nn
import torch.nn.functional as F


class DNN(nn.Module):
	def __init__(self, input_shape, output_size):
		"""
		input_shape: tuple (C, H, W)
		output_size: nombre de classes (6 dans ce projet)
		"""
		super(DNN, self).__init__()

		# Calcul de la taille d'entrée aplatie (98 * 40 = 3920 si MFCC standard)
		self.input_dim = int(input_shape[0] * input_shape[1] * input_shape[2])

		self.network = nn.Sequential(
			nn.Flatten(),

			nn.Linear(self.input_dim, 128),
			nn.ReLU(),
			nn.Dropout(p=0.5),  #

			nn.Linear(128, 128),
			nn.ReLU(),
			nn.Dropout(p=0.5),

			nn.Linear(128, 128),
			nn.ReLU(),
			nn.Dropout(p=0.5),

			nn.Linear(128, output_size)
		)

	def forward(self, x):
		return self.network(x)
