import h5py
from supplemental_code import *
import cv2
import dlib
import openface
import torch
import math
import matplotlib.pyplot as plt
import math
#from train import *





f = open("Landmarks68_model2017-1_face12_nomouth.anl", "r")
landmark_indices = []
for l in f:
	landmark_indices.append(int(l))



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

def rotate_G(G_vertices, omega, t):
	#t = np.asarray([0,0,-500])
	#print(G_vertices.type(), omega.type(), t.type())
	R = rotation_matrix(omega)
	rotated_translated_vertices1 = (G_vertices @ R.T ) + t
	return rotated_translated_vertices1.to(device)

def obtain_G_from_PCA(alphas, gammas):
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

	color_mean = np.asarray(bfm['color/model/mean'] , dtype=np.float32)
	color_mean = color_mean.reshape((-1, 3))
	triangle_topology = np.asarray(bfm['shape/representer/cells'] , dtype=np.float32)

	"""
	r1, r2 = -1, 1 # sample uniformly for bounds in U(-1,1)
	a_alpha, b_alpha = 1,30
	alphas = torch.FloatTensor(a_alpha,b_alpha).uniform_(r1, r2)

	a_gamma, b_gamma = 1,20
	gammas = torch.FloatTensor(a_gamma, b_gamma).uniform_(r1, r2)
	"""
	pca_weights_ID = torch.from_numpy(pca_weights_ID).to(device)
	pca_basis_ID = torch.from_numpy(pca_basis_ID).to(device)
	pca_var_ID = torch.sqrt(torch.from_numpy(pca_var_ID)).to(device)

	pca_weights_exp = torch.from_numpy(pca_weights_exp).to(device)
	pca_basis_exp = torch.from_numpy(pca_basis_exp).to(device)
	pca_var_exp = torch.sqrt(torch.from_numpy(pca_var_exp)).to(device)

	#print(pca_weights_ID.is_cuda,pca_basis_ID.is_cuda, alphas.is_cuda, pca_var_ID.is_cuda)
	G_vertices_1 = pca_weights_ID + torch.squeeze((pca_basis_ID @ (torch.mul(alphas, pca_var_ID)).T))
	G_vertices_2 = pca_weights_exp + torch.squeeze((pca_basis_exp @ (torch.mul(gammas, pca_var_exp)).T))

	G_vertices = G_vertices_1.to(device) + G_vertices_2.to(device)
	return G_vertices.to(device)

def get_2d_chords(G):
	img_path = "marcel_image.jfif"

	img = cv2.imread(img_path)
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

	#second_half = (P @ to_homogene(G).T)

	chords4d = torch.from_numpy(V) @ torch.from_numpy(P) @ to_homogene(G).T
	#print(second_half, V)

	chords3d = from_homogene(chords4d.T) # convert from 4d to 3d with /

	chords2d = chords3d[:,:2] # throw away depth and transpose to get coordinates in pairs

	#chords2d = np.asarray(chords2d).astype(int)
	return chords2d

device = "cpu"

alphas = torch.load("alphas.pt").cpu().detach()
gammas = torch.load("gammas.pt").cpu().detach()
omega = torch.load("omega.pt").cpu().detach()
t = torch.load("t.pt").cpu().detach()

G = obtain_G_from_PCA(alphas, gammas)
G = rotate_G(G, omega, t)
landmarks = get_2d_chords(G)

img_path = "marcel_image.jfif"

img = cv2.imread(img_path)[...,::-1]

print(img.shape, landmarks.shape)

# loop over landmarks
# 	For each coordinate e.g. (222.3, 103.7)
# 		- Interpolate the coordinates wrt to the previous and next coordinate
#		- We now have full ints per chords pair
# 		- For each whole coordinate, extract rgb value

# for interpolation i used https://en.wikipedia.org/wiki/Bilinear_interpolation
# and the figure shown in https://www.researchgate.net/figure/The-bilinear-interpolation-grid-Given-a-point-x-y-we-interpolate-between-floorx-and_fig1_221437206
d = []
for (x,y) in landmarks:
	x,y = x.item(), y.item()
	q11 = (math.floor(x), math.ceil(y))
	q12 = (math.floor(x), math.floor(y))
	q21 = (math.ceil(x), math.ceil(y))
	q22 = (math.ceil(x), math.floor(y))

	x1 = q11[0]
	x2 = q21[0]

	y1 = q11[1]
	y2 = q12[1]
	#print(x,y)
	#print(q11, q12, q21, q22)

	#print(img[q11[1],q11[0]])
	rgb_list = []
	fq_matrix = np.asarray([[img[q11[1],q11[0]],img[q12[1],q12[0]]],[img[q21[1],q21[0]], img[q22[1],q22[0]]]], dtype = float)

	for i in range(0,3):
		fq_matrix_temp = fq_matrix[:,:,i]
		#print(fq_matrix_temp)
		#print(x2,x1,y2,y1)
		#print((1/(x2 - x1)*(y2-y1)))
		interpolated = (1/(x2 - x1)*(y2-y1)) * np.asarray([x2-x, x-x1]) @ fq_matrix_temp @ np.asarray([[y2 - y],[y - y1]])
		rgb_list.append(interpolated/255.0)
	#d[(x,y)] = rgb_list
	d.append(rgb_list)

bfm = h5py.File("model2017-1_face12_nomouth.h5" , 'r' )
triangle_topology = np.asarray(bfm['shape/representer/cells'])
# uvz - matrix of shape Nx3, where N is an amount of vertices
# color - matrix of shape Nx3, where N is an amount of vertices, 3 channels represent R,G,B color scaled from 0 to 1
# triangles - matrix of shape Mx3, where M is an amount of triangles, each column represents a vertex index

print(G.shape, np.asarray(d).squeeze(2).shape, triangle_topology.shape)

print(np.asarray(d).squeeze())
print(G.numpy())
img = render(G.numpy(), np.asarray(d).squeeze(), triangle_topology.T, img.shape[0], img.shape[1])
print(img)
print(img.shape)
plt.imshow(img)
plt.show()
#print(img[landmarks].shape, img[landmarks])