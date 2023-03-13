import h5py
import numpy as np
from pprint import pprint
import torch
from fio import * # to make use of save_obj function
def get_vertices():
	bfm = h5py.File("model2017-1_face12_nomouth.h5" , 'r' )
	# Select a specific weight from BFM
	pca_weights_ID = np.asarray(bfm['shape/model/mean'] , dtype=np.float32)
	print(f"before shape: {pca_weights_ID.shape}")
	pca_weights_ID = np.reshape(pca_weights_ID, ((-1,3))) # reshape to vector to allow u_id + E_id in G
	print(f"after shape: {pca_weights_ID.shape}")
	pca_basis_ID = np.asarray(bfm['shape/model/pcaBasis'] , dtype=np.float32)[:,:30].reshape((-1,3,30)) #slice to obtain 30 pc
	pca_var_ID = np.asarray(bfm['shape/model/pcaVariance'] , dtype=np.float32)[:30]

	pca_weights_exp = np.asarray(bfm['expression/model/mean'] , dtype=np.float32)
	pca_weights_exp = np.reshape(pca_weights_exp , ((-1,3)))# reshape to vector to allow u_exp + E_exp in G
	pca_basis_exp = np.asarray(bfm['expression/model/pcaBasis'] , dtype=np.float32)[:,:20].reshape((-1,3,20)) # slice to obtain 20 pc
	pca_var_exp = np.asarray(bfm['expression/model/pcaVariance'] , dtype=np.float32)[:20]

	triangle_topology = np.asarray(bfm['shape/representer/cells'] , dtype=np.float32)
	color_mean = np.asarray(bfm['color/model/mean'] , dtype=np.float32)
	color_mean = color_mean.reshape((-1, 3))
	r1, r2 = -1, 1 # sample uniformly for bounds in U(-1,1)
	a_alpha, b_alpha = 1,30
	alphas = torch.FloatTensor(a_alpha,b_alpha).uniform_(r1, r2)

	a_gamma, b_gamma = 1,20
	gammas = torch.FloatTensor(a_gamma, b_gamma).uniform_(r1, r2)

	pca_weights_ID = torch.from_numpy(pca_weights_ID)
	pca_basis_ID = torch.from_numpy(pca_basis_ID)
	pca_var_ID = torch.from_numpy(pca_var_ID)

	pca_weights_exp = torch.from_numpy(pca_weights_exp)
	pca_basis_exp = torch.from_numpy(pca_basis_exp)
	pca_var_exp = torch.from_numpy(pca_var_exp)



	G_vertices_1 = pca_weights_ID + torch.squeeze((pca_basis_ID @ (torch.mul(alphas, pca_var_ID)).T))
	G_vertices_2 = pca_weights_exp + torch.squeeze((pca_basis_exp @ (torch.mul(gammas, pca_var_exp)).T))
	#G_vertices_2 = pca_weights_exp + pca_basis_exp @ (torch.mul(gammas, pca_var_exp.reshape(1,-1))).T

	#print(G_vertices_1.shape, G_vertices_2.shape)
	G_vertices = G_vertices_1 + G_vertices_2


	print("===")
	print(np.asarray(G_vertices).shape, color_mean.shape, triangle_topology.shape)
	# point cloud, color, triangle topology
	save_obj("temp_morpho/temp_file.obj", np.asarray(G_vertices), color_mean, triangle_topology.T) # random transpose at triangle_topolgy fixes issue
	return G_vertices, color_mean, triangle_topology

get_vertices()
