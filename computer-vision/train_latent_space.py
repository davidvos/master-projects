import h5py
from supplemental_code import *
import cv2
import dlib
import openface
import torch
import math
import matplotlib.pyplot as plt


f = open("Landmarks68_model2017-1_face12_nomouth.anl", "r")
landmark_indices = []
for l in f:
	landmark_indices.append(int(l))


img_path = "marcel_image.jfif"

img = cv2.imread(img_path)
#plt.imshow(img)
#plt.show()
def predict_landmarks(img):
	detector = dlib.get_frontal_face_detector()
	dets = detector(img, 1)
	if len(dets) < 1:
		return None # Face Not Found
		  
	print("Found %d faces" % len(dets))
	d = dets[0]
	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	landmarks = predictor.findLandmarks(gray_img, d)
	return np.asarray(landmarks)

def rotation_matrix(degrees):
	rads = degrees * (math.pi/180)
	rotation = np.asarray([[math.cos(rads), 0, math.sin(rads)], [0,1,0],[-math.sin(rads), 0, math.cos(rads)]])
	return rotation

def to_homogene(points):
	ones = np.asarray(np.ones(points.shape[0]))
	return np.column_stack((points, ones))

def from_homogene(points):

	return points[:,:points.shape[-1]-1]/ points[:,[-1]]

predictor = openface.AlignDlib("shape_predictor_68_face_landmarks.dat")
marcel_landmarks = predict_landmarks(img)
ground_truth = marcel_landmarks

bfm = h5py.File("model2017-1_face12_nomouth.h5" , 'r' )
# Select a specific weight from BFM
pca_weights_ID = np.asarray(bfm['shape/model/mean'] , dtype=np.float32)
pca_weights_ID = np.reshape(pca_weights_ID, ((-1,3))) # reshape to vector to allow u_id + E_id in G
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
pca_var_ID = torch.sqrt(torch.from_numpy(pca_var_ID))

pca_weights_exp = torch.from_numpy(pca_weights_exp)
pca_basis_exp = torch.from_numpy(pca_basis_exp)
pca_var_exp = torch.sqrt(torch.from_numpy(pca_var_exp))

G_vertices_1 = pca_weights_ID + torch.squeeze((pca_basis_ID @ (torch.mul(alphas, pca_var_ID)).T))
G_vertices_2 = pca_weights_exp + torch.squeeze((pca_basis_exp @ (torch.mul(gammas, pca_var_exp)).T))

G_vertices = G_vertices_1 + G_vertices_2

color_mean = np.asarray(bfm['color/model/mean'] , dtype=np.float32)
color_mean = color_mean.reshape((-1, 3))
triangle_topology = np.asarray(bfm['shape/representer/cells'] , dtype=np.float32)
save_obj("temp_morpho/temp_file_latent_space.obj", np.asarray(G_vertices), color_mean, triangle_topology.T)

G_vertices = G_vertices[landmark_indices]
height = img.shape[0]
width = img.shape[1]
v_r, v_l = img.shape[0], 0
v_t, v_b = img.shape[1], 0

V = np.asarray([[(v_r - v_l)/2, 0, 0, (v_r + v_l)/2],[0, (v_t-v_b)/2, 0,(v_t+v_b)/2],[0,0,.5,.5],[0,0,0,1]])


n,f = 5, 50
fov = 0.5
aspect_ratio = 1
t = np.tanh(fov/2) * n
b = -t 
r = t * aspect_ratio
l = -t * aspect_ratio
P = np.asarray([[(2*n)/(r-l), 0, (r+l)/(r-l), 0], [0, (2*n)/(t-b), (t+b)/(t-b), 0], [0,0,-(f+n)/(f-n), -(2*f*n)/(f-n)],[0,0,-1,0]])

t = np.asarray([0,0,-500])
R = rotation_matrix(10)
rotated_translated_vertices1 = (G_vertices @ R.T ) + t


second_half = (P @ to_homogene(rotated_translated_vertices1).T)

chords4d = torch.from_numpy(V) @ torch.from_numpy(P) @ to_homogene(rotated_translated_vertices1).T
print(second_half, V)

chords3d = from_homogene(chords4d.T) # convert from 4d to 3d with /

chords2d = chords3d[:,:2] # throw away depth and transpose to get coordinates in pairs

chords2d = np.asarray(chords2d).astype(int)

print(rotated_translated_vertices1)
x = [x.item() for (x,_) in chords2d]
y = [y.item() for (_,y) in chords2d]
plt.plot(x,y, marker = '.', linestyle = 'None')
plt.show()

