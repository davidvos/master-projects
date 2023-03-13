import h5py
import numpy as np
from pprint import pprint
import torch
from fio import * # to make use of save_obj function
from morphable_model_play_lau import *
import math


def read_anl_file(file_path):
	with open(file_path, "r") as f:
		landmarks = [int(landmark[:-1]) for landmark in f]
	return landmarks



def rotation_matrix(degrees):
	rads = degrees * (math.pi/180)
	rotation = np.asarray([[math.cos(rads), 0, math.sin(rads)], [0,1,0],[-math.sin(rads), 0, math.cos(rads)]])
	return rotation

def only_rotate(G_vertices, color_mean, triangle_topology):



	"""
	rotation_matrix1 = rotatation_matrix(-10)
	rotated_G_vertices = G_vertices @ rotation_matrix1

	print(f"shape vertices: {G_vertices.shape}")
	print(f"shape rotation_matrix: {rotation_matrix1.shape}")
	print(f"shape result: {rotated_G_vertices.shape}")
	"""
	OG = G_vertices
	rotated_vertices1 = G_vertices @ rotation_matrix(10)
	rotated_vertices2 = G_vertices @ rotation_matrix(-10)

	save_obj("temp_morpho/og_file.obj", np.asarray(OG), color_mean, triangle_topology.T) # random transpose at triangle_topolgy fixes issue
	save_obj("temp_morpho/plusRotated.obj", np.asarray(rotated_vertices1), color_mean, triangle_topology.T)
	save_obj("temp_morpho/negRotated.obj", np.asarray(rotated_vertices2), color_mean, triangle_topology.T)

def rotate_and_translate(G_vertices, color_mean, triangle_topology):
	OG = G_vertices
	t = np.asarray([0,0,-500])
	rotated_translated_vertices1 = (G_vertices @ rotation_matrix(10)) + t

	save_obj("temp_morpho/og_file.obj", np.asarray(OG), color_mean, triangle_topology.T) # random transpose at triangle_topolgy fixes issue
	save_obj("temp_morpho/plusRotatedAndTranslated.obj", np.asarray(rotated_translated_vertices1), color_mean, triangle_topology.T)


G_vertices, color_mean, triangle_topology = get_vertices()
file_path = "Landmarks68_model2017-1_face12_nomouth.anl"
landmarks = read_anl_file(file_path)

only_rotate()
rotate_and_translate(G_vertices, color_mean, triangle_topology)