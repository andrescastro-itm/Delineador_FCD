import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

from unet import load_model, segment
from cnn import load_model_clas

from get_data import preprocess


def inference(config):
	#Leer volumen
	input_path = config['files']['input']
	vol = preprocess(input_path,config, norm = True)
	print(vol.shape)

	#Crear Carpeta de salida en caso que no exista
	out_path = config['files']['output']
	if not os.path.exists(out_path):
		os.makedirs(out_path)

	#Cargar modelo clasificación
	cnn = load_model_clas(config)
	cnn.eval()

	#Cargar modelo delineación
	unet = load_model(config)
	unet.eval()
	#print(unet)

	name_study = input_path.split('/')[-1].split('.')[0]

	#Iterar por slices en vista axial
	if config['numslice'] == -1:
	
		for i in tqdm(range(vol.shape[2]), bar_format='Slices procesados: {desc}{percentage:3.0f}%|{bar:30}|'):
			slice = torch.tensor(vol[...,i]).unsqueeze(0).unsqueeze(1)
			prob = cnn(slice)
			print(f'salida red clasificación: {prob.item()}')
			if config['useclas']:
				if (prob >= 0.5):
					plot = True
				else:
					plot = False
			else:
				plot=True

			if plot:		
				seg = segment(unet, slice)
				fig = plt.figure(figsize=(128*4/300, 128*4/300),frameon=False)
				ax = plt.Axes(fig, [0., 0., 1., 1.])
				ax.set_axis_off()
				fig.add_axes(ax)
				ax.imshow(vol[...,i], cmap='gray')
				seg = np.where(seg==0., np.nan,1.)
				ax.imshow(seg, cmap='Oranges_r', alpha=0.7)
				fig.savefig(f'{out_path}{name_study}_slice_{i}.png', dpi=300)
				plt.close()

	else:
		i = config['numslice']
		slice = torch.tensor(vol[...,i]).unsqueeze(0).unsqueeze(1)
		seg = segment(unet, slice)
		fig = plt.figure(figsize=(128*4/300, 128*4/300),frameon=False)
		ax = plt.Axes(fig, [0., 0., 1., 1.])
		ax.set_axis_off()
		fig.add_axes(ax)
		ax.imshow(vol[...,i], cmap='gray')
		seg = np.where(seg==0., np.nan,1.)
		ax.imshow(seg, cmap='Oranges_r', alpha=0.7)
		fig.savefig(f'{out_path}{name_study}_slice_{i}.png', dpi=300)
		plt.close()

