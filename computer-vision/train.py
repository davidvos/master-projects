import h5py
from supplemental_code import *
import cv2
import dlib
import openface
import torch
import math
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.optim as optim
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
"""f = open("Landmarks68_model2017-1_face12_nomouth.anl", "r")
landmark_indices = []
for l in f:
	landmark_indices.append(int(l))"""


img_path = "marcel_image.jfif"

img = cv2.imread(img_path)
#plt.imshow(img)
#plt.show()
def predict_landmarks(img):
	predictor = openface.AlignDlib("shape_predictor_68_face_landmarks.dat")
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
	rotation = torch.from_numpy(np.asarray([[math.cos(rads), 0, math.sin(rads)], [0,1,0],[-math.sin(rads), 0, math.cos(rads)]])).to(device)
	return rotation

def to_homogene(points):
	#ones = np.asarray(np.ones(points.shape[0]))
	#return torch.from_numpy(np.column_stack((points, ones)))
	ones = torch.ones(points.shape[0],1).to(device)
	#print(points.is_cuda, )
	return(torch.cat((points, ones), 1)).double().to(device)

def from_homogene(points):
	return points[:,:points.shape[-1]-1]/ points[:,[-1]].to(device)

def rotate_G(G_vertices, omega, t):
	#t = np.asarray([0,0,-500])
	#print(G_vertices.type(), omega.type(), t.type())
	R = rotation_matrix(omega).float().to(device)
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


def manual_calculation_landmarks(G_vertices):
	landmark_indices = get_landmark_indices()
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

	#second_half = (P @ to_homogene(G_vertices).T)

	chords4d = torch.from_numpy(V).to(device) @ torch.from_numpy(P).to(device) @ (to_homogene(G_vertices).T).to(device)

	chords3d = from_homogene(chords4d.T).to(device) # convert from 4d to 3d with /

	chords2d = chords3d[:,:2] # throw away depth and transpose to get coordinates in pairs

	#chords2d = torch.from_numpy(np.asarray(chords2d).astype(int))
	landmarks = chords2d
	return landmarks

def get_landmark_indices():
	f = open("Landmarks68_model2017-1_face12_nomouth.anl", "r")
	landmark_indices = []
	for l in f:
		landmark_indices.append(int(l))
	return landmark_indices

def visualise_estimated_landmarks(chords2d):
	x = [x.item() for (x,_) in chords2d]
	y = [y.item() for (_,y) in chords2d]
	plt.plot(x,y, marker = '.', linestyle = 'None')
	plt.show()

def obtain_ground_truth(img_path):
	img = cv2.imread(img_path)
	img = cv2.flip(img, 1)
	#print(img[0:,:,])
	#quit(1)
	#img[,::] += max(img[,:,:]) 
	#plt.imshow(img[..., ::-1])
	#plt.show()
	predictor = openface.AlignDlib("shape_predictor_68_face_landmarks.dat")
	marcel_landmarks = predict_landmarks(img)
	#print(marcel_landmarks.shape)
	#quit(1)
	ground_truth = marcel_landmarks
	return ground_truth

def train_loop():


	n_iter = 10**4
	#n_iter = 3000
	img_path = "marcel_image.jfif"
	y_labels = torch.from_numpy(obtain_ground_truth(img_path)).to(device)
	y_labels[:,1] = -y_labels[:,1] + y_labels[:,1].max()

	alphas = Variable(torch.ones(30,)).to(device).detach().requires_grad_(True)
	gammas  = Variable(torch.ones(20,)).to(device).detach().requires_grad_(True)
	omega = Variable(torch.ones(1,)).to(device).detach().requires_grad_(True)
	t = Variable(torch.as_tensor(np.asarray([0.0,0.0,-500.0]))).to(device).detach().requires_grad_(True)

	#t[2] = -500 \
	#print(alphas.is_cuda)
	optimizer = optim.Adam([alphas, gammas, omega, t], lr=0.001)
	
	losses = []
	loss_lan = nn.MSELoss()

	lambda_alpha, lambda_delta = 1000, 1000
	eval_every = 100
	first_iter = True
	for i in range(n_iter):
		optimizer.zero_grad()
		G = obtain_G_from_PCA(alphas, gammas)
		G = rotate_G(G, omega, t)
		predicted_landmarks = manual_calculation_landmarks(G)
		if first_iter:
			first_predicted_landmarks = predicted_landmarks
			first_iter = False
		out_loss_lan = loss_lan(predicted_landmarks.double(), y_labels.double())
		out_loss_reg = lambda_alpha * (alphas**2).sum() + lambda_delta * (gammas**2).sum()
		total_loss = (out_loss_lan.double() + out_loss_reg.double()).double()
		#print(total_loss.type())
		total_loss.backward()
		optimizer.step()
		losses.append(total_loss)

		if i % eval_every == 0:
			print(F"Iteration: {i}/{n_iter} with loss: {total_loss}")
			#print(f"==total_loss: {total_loss}")
			#if i % 1000 == 0:
				#visualise_estimated_landmarks(predicted_landmarks)

	torch.save(alphas, 'alphas2.pt')
	torch.save(gammas, 'gammas2.pt')
	torch.save(omega, 'omega2.pt')
	torch.save(t, 't2.pt')
	#visualise_estimated_landmarks(predicted_landmarks)
	return predicted_landmarks, first_predicted_landmarks

"""
predicted_landmarks, first_predicted_landmarks = train_loop()
predicted_landmarks = predicted_landmarks.cpu().detach()
first_predicted_landmarks = first_predicted_landmarks.cpu().detach()
marcel_landmarks = obtain_ground_truth(img_path)
marcel_landmarks[:,1] = -marcel_landmarks[:,1] + marcel_landmarks[:,1].max()
for index, (pred, true_label) in enumerate(zip(predicted_landmarks, marcel_landmarks)):
	print(pred, true_label, first_predicted_landmarks[index])
# print(-marcel_landmarks[:,0])
#marcel_landmarks[:,1] = -marcel_landmarks[:,1] + marcel_landmarks[:,1].max()
# quit(1)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.suptitle('Horizontally stacked subplots')
ax1.plot([x.item() for (x,_) in predicted_landmarks], [y.item() for (_,y) in predicted_landmarks], marker = '.', linestyle = 'None')
ax2.plot([x for (x,_) in marcel_landmarks], [y for (_,y) in marcel_landmarks], marker = '.', linestyle = 'None')
ax3.plot([x.item() for (x,_) in first_predicted_landmarks], [y.item() for (_,y) in first_predicted_landmarks], marker = '.', linestyle = 'None')
plt.show()"""







