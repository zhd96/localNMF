import os
import sys
import mmap
import cv2

import matplotlib.pyplot as plt
import pylab
import networkx as nx
import numpy as np
import scipy.io as sio
import scipy.stats as ss
import scipy.ndimage
import scipy.signal

from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import NMF
from sklearn import linear_model
from scipy.ndimage.filters import convolve

def local_correlations_fft(Y, eight_neighbours=True, swap_dim=True, opencv=True):
    """Computes the correlation image for the input dataset Y using a faster FFT based method, adapt from caiman
    Parameters:
    -----------
    Y:  np.ndarray (3D or 4D)
        Input movie data in 3D or 4D format
    eight_neighbours: Boolean
        Use 8 neighbors if true, and 4 if false for 3D data (default = True)
        Use 6 neighbors for 4D data, irrespectively
    swap_dim: Boolean
        True indicates that time is listed in the last axis of Y (matlab format)
        and moves it in the front
    opencv: Boolean
        If True process using open cv method
    Returns:
    --------
    Cn: d1 x d2 [x d3] matrix, cross-correlation with adjacent pixels
    """

    if swap_dim:
        Y = np.transpose(
            Y, tuple(np.hstack((Y.ndim - 1, list(range(Y.ndim))[:-1]))))

    Y = Y.astype('float32')
    Y -= np.mean(Y, axis=0)
    Ystd = np.std(Y, axis=0)
    Ystd[Ystd == 0] = np.inf
    Y /= Ystd

    if Y.ndim == 4:
        if eight_neighbours:
            sz = np.ones((3, 3, 3), dtype='float32')
            sz[1, 1, 1] = 0
        else:
            sz = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                           [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                           [[0, 0, 0], [0, 1, 0], [0, 0, 0]]], dtype='float32')
    else:
        if eight_neighbours:
            sz = np.ones((3, 3), dtype='float32')
            sz[1, 1] = 0
        else:
            sz = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype='float32')

    if opencv and Y.ndim == 3:
        Yconv = Y.copy()
        for idx, img in enumerate(Yconv):
            Yconv[idx] = cv2.filter2D(img, -1, sz, borderType=0)
        MASK = cv2.filter2D(
            np.ones(Y.shape[1:], dtype='float32'), -1, sz, borderType=0)
    else:
        Yconv = convolve(Y, sz[np.newaxis, :], mode='constant')
        MASK = convolve(
            np.ones(Y.shape[1:], dtype='float32'), sz, mode='constant')
    Cn = np.mean(Yconv * Y, axis=0) / MASK
    return Cn

def noise_estimator(Y,range_ff=[0.25,0.5],method='logmexp'):
	"""
	estimate noise level for each pixel, adapt from kelly's code
	Parameters:
	----------------
	Y: np.darray(2d or 3d): dimension d x T or d1 x d2 x T

	Return:
	----------------
	sns: np.darray(2d): dimension d x 1 or d1 x d2
		standard deviation of noise

	""" 
	dims = Y.shape
	if len(dims)>2:
	    V_hat = Y.reshape((np.prod(dims[:2]),dims[2]),order='F')
	else:
	    V_hat = Y.copy()
	sns = []
	for i in range(V_hat.shape[0]):
		ff, Pxx = scipy.signal.welch(V_hat[i,:],nperseg=min(256,dims[-1]))
		ind1 = ff > range_ff[0]
		ind2 = ff < range_ff[1]
		ind = np.logical_and(ind1, ind2)
		#Pls.append(Pxx)
		#ffs.append(ff)
		Pxx_ind = Pxx[ind]
		sn = {
		    'mean': lambda Pxx_ind: np.sqrt(np.mean(np.divide(Pxx_ind, 2))),
		    'median': lambda Pxx_ind: np.sqrt(np.median(np.divide(Pxx_ind, 2))),
		    'logmexp': lambda Pxx_ind: np.sqrt(np.exp(np.mean(np.log(np.divide(Pxx_ind, 2)))))
		}[method](Pxx_ind)
		sns.append(sn)
	
	sns = np.asarray(sns)
	if len(dims)>2:
	    sns = sns.reshape(dims[:2],order='F')
	return sns

def show_img(ax, img,vmin=None,vmax=None):
	# Visualize local correlation, adapt from kelly's code
    im = ax.imshow(img,cmap='jet')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.1)
    if np.abs(img.min())< 1:
        format_tile ='%.2f'
    else:
        format_tile ='%5d'
    plt.colorbar(im, cax=cax,orientation='horizontal',
                 spacing='uniform')

def threshold_data(Yd, th=2):
	"""
	Threshold data: in each pixel, compute the median and median absolute deviation (MAD), 
	then zero all bins (x,t) such that Yd(x,t) < med(x) + th * MAD(x).  Default value of th is 2. 
 
	Parameters:
	----------------
	Yd: 3d np.darray: dimension d1 x d2 x T
		denoised data

	Return:
	----------------
	Yt: 3d np.darray: dimension d1 x d2 x T
		cleaned, thresholded data

	""" 
	dims = Yd.shape;
	Yt = np.zeros(dims);
	ii=0;
	for array in [Yd]:
	    Yd_median = np.median(array, axis=2, keepdims=True)
	    Yd_mad = np.median(abs(array - Yd_median), axis=2, keepdims=True)
	    for i in range(dims[2]):
	        Yt[:,:,i] = np.clip(array[:,:,i], a_min = (Yd_median + th*Yd_mad)[:,:,0], a_max = None) - (Yd_median + th*Yd_mad)[:,:,0]
    
	return Yt

def find_superpixel(Yt, cut_off_point, length_cut, eight_neighbours=True):
	"""
	Find superpixels in Yt.  For each pixel, calculate its correlation with neighborhood pixels.  
	If it's larger than threshold, we connect them together.  In this way, we form a lot of connected components.
	If its length is larger than threshold, we keep it as a superpixel.

	Parameters:
	----------------
	Yt: 3d np.darray, dimension d1 x d2 x T
		thresholded data
	cut_off_point: double scalar
		correlation threshold
	length_cut: double scalar
		length threshold
	eight_neighbours: Boolean
		Use 8 neighbors if true.  Defalut value is True.
	
	Return:
	----------------
	connect_mat_1: 2d np.darray, d1 x (d2*num_plane)
		illustrate position of each superpixel.  
		Each superpixel has a random number "indicator".  Same number means same superpixel.

	idx-1: double scalar
		number of superpixels
	
	comps: list, length = number of superpixels
		comp on comps is also list, its value is position of each superpixel in Yt_r = Yt.reshape(np.prod(dims[:2]),-1,order="F")

	permute_col: list, length = number of superpixels
		all the random numbers used to idicate superpixels in connect_mat_1

	""" 

	dims = Yt.shape;
	ref_mat = np.arange(np.prod(dims[:-1])).reshape(dims[:-1],order='F')
	######### calculate correlation ############
	w_mov = (Yt.transpose(2,0,1) - np.mean(Yt, axis=2)) / np.std(Yt, axis=2);
	w_mov[np.isnan(w_mov)] = 0;
	
	rho_v = np.mean(np.multiply(w_mov[:, :-1, :], w_mov[:, 1:, :]), axis=0)
	rho_h = np.mean(np.multiply(w_mov[:, :, :-1], w_mov[:, :, 1:]), axis=0)
	
	eight_neighbours = True;
	
	if eight_neighbours:
	    rho_l = np.mean(np.multiply(w_mov[:, 1:, :-1], w_mov[:, :-1, 1:]), axis=0)
	    rho_r = np.mean(np.multiply(w_mov[:, :-1, :-1], w_mov[:, 1:, 1:]), axis=0)
	
	rho_v = np.concatenate([rho_v, np.zeros([1, rho_v.shape[1]])], axis=0)
	rho_h = np.concatenate([rho_h, np.zeros([rho_h.shape[0],1])], axis=1)
	if eight_neighbours:
	    rho_r = np.concatenate([rho_r, np.zeros([rho_r.shape[0],1])], axis=1)
	    rho_r = np.concatenate([rho_r, np.zeros([1, rho_r.shape[1]])], axis=0)
	    rho_l = np.concatenate([np.zeros([rho_l.shape[0],1]), rho_l], axis=1)
	    rho_l = np.concatenate([rho_l, np.zeros([1, rho_l.shape[1]])], axis=0)

	################## find pairs where correlation above threshold 
	temp_v = np.where(rho_v > cut_off_point);
	A_v = ref_mat[temp_v];
	B_v = ref_mat[(temp_v[0] + 1, temp_v[1])]
	
	temp_h = np.where(rho_h > cut_off_point);
	A_h = ref_mat[temp_h];
	B_h = ref_mat[(temp_h[0], temp_h[1] + 1)]
	
	if eight_neighbours:
	    temp_l = np.where(rho_l > cut_off_point);
	    A_l = ref_mat[temp_l];
	    B_l = ref_mat[(temp_l[0] + 1, temp_l[1] - 1)]
	    
	    temp_r = np.where(rho_r > cut_off_point);
	    A_r = ref_mat[temp_r];
	    B_r = ref_mat[(temp_r[0] + 1, temp_r[1] + 1)]
	
	    A = np.concatenate([A_v,A_h,A_l,A_r])
	    B = np.concatenate([B_v,B_h,B_l,B_r])
	else:
	    A = np.concatenate([A_v,A_h])
	    B = np.concatenate([B_v,B_h])

	########### form connected componnents #########  
	G = nx.Graph()
	G.add_edges_from(list(zip(A, B)))
	comps=list(nx.connected_components(G))
	
	connect_mat=np.zeros(np.prod(dims[:2]));
	idx=1;
	for comp in comps:
	    if(len(comp) > length_cut):
	        idx = idx+1;
	
	permute_col = np.random.permutation(idx)+1;
	
	ii=0;
	for comp in comps:
	    if(len(comp) > length_cut):
	        connect_mat[list(comp)] = permute_col[ii];
	        ii = ii+1;
	connect_mat_1 = connect_mat.reshape(dims[0],dims[1],order='F');
	return connect_mat_1, idx-1, comps, permute_col

def find_superpixel_3d(Yt, num_plane, cut_off_point, length_cut, eight_neighbours=True):
	"""
	Find 3d supervoxels in Yt.  For each pixel, calculate its correlation with neighborhood pixels.  
	If it's larger than threshold, we connect them together.  In this way, we form a lot of connected components.
	If its length is larger than threshold, we keep it as a superpixel.

	Parameters:
	----------------
	Yt: 3d np.darray, dimension d1 x (d2*num_plane) x T
		thresholded data
	cut_off_point: double scalar
		correlation threshold
	length_cut: double scalar
		length threshold
	eight_neighbours: Boolean
		Use 8 neighbors in same plane if true.  Defalut value is True.
	
	Return:
	----------------
	connect_mat_1: 2d np.darray, d1 x (d2*num_plane)
		illustrate position of each superpixel.  
		Each superpixel has a random number "indicator".  Same number means same superpixel.

	idx-1: double scalar
		number of superpixels
	
	comps: list, length = number of superpixels
		comp on comps is also list, its value is position of each superpixel in Yt_r = Yt.reshape(np.prod(dims[:2]),-1,order="F")

	permute_col: list, length = number of superpixels
		all the random numbers used to idicate superpixels in connect_mat_1

	""" 	
	dims = Yt.shape;
	Yt_3d = Yt.reshape(dims[0],int(dims[1]/num_plane),num_plane,dims[2],order="F");
	dims = Yt_3d.shape;
	ref_mat = np.arange(np.prod(dims[:-1])).reshape(dims[:-1],order='F');
	######### calculate correlation ############
	w_mov = (Yt_3d.transpose(3,0,1,2) - np.mean(Yt_3d, axis=3)) / np.std(Yt_3d, axis=3);
	w_mov[np.isnan(w_mov)] = 0;
	
	rho_v = np.mean(np.multiply(w_mov[:, :-1, :], w_mov[:, 1:, :]), axis=0)
	rho_h = np.mean(np.multiply(w_mov[:, :, :-1], w_mov[:, :, 1:]), axis=0)
	
	if eight_neighbours:
		rho_l = np.mean(np.multiply(w_mov[:, 1:, :-1], w_mov[:, :-1, 1:]), axis=0)
		rho_r = np.mean(np.multiply(w_mov[:, :-1, :-1], w_mov[:, 1:, 1:]), axis=0)
	    
	rho_u = np.mean(np.multiply(w_mov[:, :, :, :-1], w_mov[:, :, :, 1:]), axis=0)
	
	rho_v = np.concatenate([rho_v, np.zeros([1, rho_v.shape[1],num_plane])], axis=0)
	rho_h = np.concatenate([rho_h, np.zeros([rho_h.shape[0],1,num_plane])], axis=1)
	if eight_neighbours:
		rho_r = np.concatenate([rho_r, np.zeros([rho_r.shape[0],1,num_plane])], axis=1)
		rho_r = np.concatenate([rho_r, np.zeros([1, rho_r.shape[1],num_plane])], axis=0)
		rho_l = np.concatenate([np.zeros([rho_l.shape[0],1,num_plane]), rho_l], axis=1)
		rho_l = np.concatenate([rho_l, np.zeros([1, rho_l.shape[1],num_plane])], axis=0)
	rho_u = np.concatenate([rho_u, np.zeros([rho_u.shape[0], rho_u.shape[1],1])], axis=2)
	################## find pairs where correlation above threshold 
	temp_v = np.where(rho_v > cut_off_point);
	A_v = ref_mat[temp_v];
	B_v = ref_mat[(temp_v[0] + 1, temp_v[1], temp_v[2])]
	
	temp_h = np.where(rho_h > cut_off_point);
	A_h = ref_mat[temp_h];
	B_h = ref_mat[(temp_h[0], temp_h[1] + 1, temp_h[2])]
	
	temp_u = np.where(rho_u > cut_off_point);
	A_u = ref_mat[temp_u];
	B_u = ref_mat[(temp_u[0], temp_u[1], temp_u[2]+1)]
	
	if eight_neighbours:
		temp_l = np.where(rho_l > cut_off_point);
		A_l = ref_mat[temp_l];
		B_l = ref_mat[(temp_l[0] + 1, temp_l[1] - 1, temp_l[2])]
		
		temp_r = np.where(rho_r > cut_off_point);
		A_r = ref_mat[temp_r];
		B_r = ref_mat[(temp_r[0] + 1, temp_r[1] + 1, temp_r[2])]
	
		A = np.concatenate([A_v,A_h,A_l,A_r,A_u])
		B = np.concatenate([B_v,B_h,B_l,B_r,B_u])
	else:
		A = np.concatenate([A_v,A_h,A_u])
		B = np.concatenate([B_v,B_h,B_u])    
	########### form connected componnents #########  
	G = nx.Graph()
	G.add_edges_from(list(zip(A, B)))
	comps=list(nx.connected_components(G))
	
	connect_mat=np.zeros(np.prod(dims[:-1]));
	idx=1;
	for comp in comps:
		if(len(comp) > length_cut):
			idx = idx+1;
	
	permute_col = np.random.permutation(idx)+1;
	
	ii=0;
	for comp in comps:
		if(len(comp) > length_cut):
			connect_mat[list(comp)] = permute_col[ii];
			ii = ii+1;
	connect_mat_1 = connect_mat.reshape(dims[:-1],order='F');
	return connect_mat_1, idx-1, comps, permute_col


def spatial_temporal_ini(Yt, comps, idx, length_cut, method='svd', maxiter=5, whole_data=True):
	"""
	Find spatial and temporal initialization for each superpixel in Yt.  

	Parameters:
	----------------
	Yt: 3d np.darray, dimension d1 x d2 x T
		thresholded data
	comps: list, length = number of superpixels
		position of each superpixel
	idx: double scalar
		number of superpixels
	length_cut: double scalar
		length threshold
	method: string, "svd" or "iterate"
		"svd" is to do rank-1 svd for each superpixel. Default value is "svd".
		"iterate" adds background component b(x), and iterate sereval times to find rank-1 factorization of each superpixel
	maxiter: int scalar
		maximum number of iteration of method: iterate
	whole_data: Boolean
		Use whole data if True or just above threshold data to do initilization.  Default is True.

	Return:
	----------------
	V_mat: 2d np.darray, dimension T x number of superpixel
		temporal initilization
	U_mat: 2d np.darray, dimension (d1*d2) x number of superpixel
		spatial initilization
	B_mat: 2d np.darray, dimension (d1*d2) x 1
		background initilization.  Zero matrix if choose "svd" method.
	""" 

	dims = Yt.shape;
	T = dims[2];
	Yt_r= Yt.reshape(np.prod(dims[:2]),T,order = "F");
	ii = 0;
	maxiter=5;
	U_mat = np.zeros([np.prod(dims[:2]),idx]);
	B_mat = np.zeros([np.prod(dims[:2]),idx]);
	V_mat = np.zeros([T,idx]);

	if method == 'svd':
		for comp in comps:
			if(len(comp) > length_cut):
				y_temp = Yt_r[list(comp),:];
				unique_t = np.unique(np.where(y_temp > 0)[1]);
				model = NMF(n_components=1, init='random', random_state=0)
				U_mat[list(comp),ii] = model.fit_transform(y_temp)[:,0];
				V_mat[:,ii] = model.components_;
				ii = ii+1;
	elif method == 'iterate':
		for comp in comps:
			if(len(comp) > length_cut):
				y_temp = Yt_r[list(comp),:];
				unique_t = np.unique(np.where(y_temp > 0)[1]);
				b = np.median(y_temp, axis=1, keepdims=True);
				_, _, c = np.linalg.svd(y_temp - b, full_matrices=False)
				c = c[0,:].reshape(y_temp.shape[1],1);
				if (c[np.where(abs(c)==abs(c).max())[0]] < 0):
				    c=-1*c;
				c = np.maximum(0, c);
				a = np.zeros([len(comp),1]);
				f = np.ones([y_temp.shape[1],1]);
	
				mask_ab = np.ones([a.shape[0],2]);
				mask_c = np.ones([c.shape[0],1]);
	
				if whole_data:
				    ind = np.ones(y_temp.shape);
				else:
				    ind = (y_temp > 0);
				y_temp = y_temp*ind; ########### in case y0 doesn't satisfy sub-threshold data = 0 ##############
	
				for jj in range(maxiter):
				    temp = ls_solve(np.hstack((c,f)), y_temp.T, mask_ab.T, ind.T).T;
				    a = temp[:,:-1];
				    b = temp[:,[-1]];
				    c = ls_solve(a, y_temp-b, mask_c.T, ind).T;
	
				U_mat[[list(comp)],[ii]] = a.T;
				B_mat[[list(comp)],[ii]] = b.T;
				V_mat[:,[ii]] = c;
				ii = ii+1;
	return V_mat, U_mat, B_mat

def vcorrcoef(X,y):
	"""
	Calculate correlation between X array and Y vector.

	Parameters:
	----------------
	X: 2d np.darray, dimension T x d
	Y: 2d np.darray, dimension T x 1

	Return:
	----------------
	r: 2d np.darray, dimension d x 1
	correlation vector
	""" 

	Xm = np.reshape(np.mean(X,axis=1),(X.shape[0],1))
	ym = np.mean(y)
	r_num = np.sum((X-Xm)*(y-ym),axis=1)
	r_den = np.sqrt(np.sum((X-Xm)**2,axis=1)*np.sum((y-ym)**2))
	r = r_num/r_den
	return r

def search_superpixel_in_range(x_range,y_range, connect_mat, permute_col, V_mat):
	"""
	Search superpixel within a defined range

	Parameters:
	----------------
	x_range: list, length = 2
		vertical range: [up, down], include both up and down rows.
	y_range: list, length = 2
		horizonal range: [left, right], include both left and right columns.
	connect_mat_1: 2d np.darray, d1 x d2
		illustrate position of each superpixel, same value means same superpixel
	permute_col: list, length = number of superpixels
		random number used to idicate superpixels in connect_mat_1
	V_mat: 2d np.darray, dimension T x number of superpixel
		temporal initilization

	Return:
	----------------
	unique_pix: list, length idx (number of superpixels)
		random numbers for superpixels in this patch
	M: 2d np.array, dimension T x idx
		temporal components for superpixels in this patch
	""" 

	unique_pix = np.sort(np.unique(connect_mat[x_range[0]:(x_range[1]+1), y_range[0]:(y_range[1]+1)]))[1:];
	M = np.zeros([V_mat.shape[0], len(unique_pix)]);
	for ii in range(len(unique_pix)):
	    M[:,ii] =  V_mat[:,int(np.where(permute_col==unique_pix[ii])[0])];

	return unique_pix, M

def fast_sep_nmf(M, r, th, normalize=1):
	"""
	Find pure superpixels. solve nmf problem M = M(:,K)H, K is a subset of M's columns.

	Parameters:
	----------------
	M: 2d np.array, dimension T x idx
		temporal components of superpixels.
	r: int scalar
		maximum number of pure superpixels you want to find.  Usually it's set to idx, which is number of superpixels.
	th: double scalar, correlation threshold
		Won't pick up two pure superpixels, which have correlation higher than th.
	normalize: Boolean.
		Normalize L1 norm of each column to 1 if True.  Default is True.

	Return:
	----------------
	pure_pixels: 1d np.darray, dimension d x 1. (d is number of pure superpixels)
		pure superpixels for these superpixels, actually column indices of M.
	coef: 2d np.darray, dimension d x idx.
		coefficient matrix H.
	coef_rank: 2d np.darray, dimension d x idx.
		each superpixel is expalined by which pure superpixel most.
	""" 

	pure_pixels = [];
	if normalize == 1:
		M = M/np.sum(M, axis=0,keepdims=True);

	normM = np.sum(M**2, axis=0,keepdims=True);
	normM_orig = normM.copy();
	normM_sqrt = np.sqrt(normM);
	nM = np.sqrt(normM);
	ii = 0;
	U = np.zeros([M.shape[0], r]);
	while ii < r and (normM_sqrt/nM).max() > th:
		## select the column of M with largest relative l2-norm
		temp = normM/normM_orig;
		pos = np.where(temp == temp.max())[1][0];
		## check ties up to 1e-6 precision
		pos_ties = np.where((temp.max() - temp)/temp.max() <= 1e-6)[1];
		if len(pos_ties) > 1:
			pos = pos_ties[np.where(normM_orig[0,pos_ties] == (normM_orig[0,pos_ties]).max())[0][0]];

		## update the index set, and extracted column
		pure_pixels.append(pos);
		U[:,ii] = M[:,pos];
		for jj in range(pos):
			U[:,ii] = U[:,ii] - U[:,jj]*sum(U[:,jj]*U[:,ii])

		U[:,ii] = U[:,ii]/np.sqrt(sum(U[:,ii]**2));
		normM = np.maximum(0, normM - np.matmul(U[:,[ii]].T, M)**2);
		normM_sqrt = np.sqrt(normM);
		ii = ii+1;
	coef = np.matmul(np.matmul(np.linalg.inv(np.matmul(M[:,pure_pixels].T, M[:,pure_pixels])), M[:,pure_pixels].T), M);
	pure_pixels = np.array(pure_pixels);
	coef_rank = coef.copy(); ##### from large to small 
	for ii in range(len(pure_pixels)):
		coef_rank[:,ii] = [x for _,x in sorted(zip(len(pure_pixels) - ss.rankdata(coef[:,ii]), pure_pixels))];
	return pure_pixels, coef, coef_rank

def prepare_iteration(Yt, connect_mat_1, permute_col, unique_pix, pure_pixels, V_mat, U_mat, x_range, y_range, num_plane=1):
	"""
	Get some needed variables for the successive nmf iterations.

	Parameters:
	----------------
	Yt: 3d np.darray, dimension d1 x d2 x T
		thresholded data
	connect_mat_1: 2d np.darray, d1 x d2
		illustrate position of each superpixel, same value means same superpixel
	permute_col: list, length = number of superpixels
		random number used to idicate superpixels in connect_mat_1
	unique_pix: 2d np.darray, dimension d x 1
		random numbers for superpixels in this patch
	pure_pixels: 1d np.darray, dimension d x 1. (d is number of pure superpixels)
		pure superpixels for these superpixels, actually column indices of M.	
	V_mat: 2d np.darray, dimension T x number of superpixel
		temporal initilization
	U_mat: 2d np.darray, dimension (d1*d2) x number of superpixel
		spatial initilization
	x_range: list, length = 2
		vertical range: [up, down], include both up and down rows.
	y_range: list, length = 2
		horizonal range: [left, right], include both left and right columns.
	num_plane: int scalar. Default is 1.

	Return:
	----------------
	a_ini: 2d np.darray, number pixels x number of pure superpixels
		initialization of spatial components
	c_ini: 2d np.darray, T x number of pure superpixels
		initialization of temporal components
	y0: 2d np.darray: number pixels x T
		threshold data for this patch
	brightness_rank: 2d np.darray, dimension d x 1
		brightness rank for pure superpixels in this patch. Rank 1 means the brightest.
	pure_pix: 2d np.darray, dimension d x 1
		random numbers for pure superpixels in this patch
	corr_img_all_r: 3d np.darray, d1 x d2 x number of pure superpixels
		correlation image: corr(y0, c_ini).
	""" 

	dims = Yt.shape;
	T = dims[2];
	if num_plane > 1:
		print("3D data!")
		Yt = Yt.reshape(dims[0],int(dims[1]/num_plane),num_plane,T, order="F");
	dims = Yt.shape;
	up = x_range[0];
	down = x_range[1];
	left = y_range[0];
	right = y_range[1];
	####################### order superpixel according to brightness ############################
	pure_pix = np.zeros(len(pure_pixels));
	brightness = np.zeros(len(pure_pixels));
	for ii in range(len(pure_pixels)):
	    pure_pix[ii] = unique_pix[pure_pixels[ii]];
	    v_max = V_mat[:,np.where(permute_col==unique_pix[pure_pixels[ii]])[0][0]].max();
	    u_max = U_mat[:,np.where(permute_col==unique_pix[pure_pixels[ii]])[0][0]].max();
	    brightness[ii] = u_max * v_max;
	brightness_rank = len(pure_pix) - ss.rankdata(brightness,method="ordinal")+1;
		####################### sort temporal trace and superpixel position according to brightness ###################
	c_ini = np.zeros([T,len(pure_pix)]);
	a_ini = np.zeros([(down-up+1)*(right-left+1)*num_plane,len(pure_pix)]);
	for ii in range(len(pure_pix)):
	    pos = np.where(permute_col==pure_pix[np.where(brightness_rank == ii+1)[0]])[0][0];
	    c_ini[:,ii] = V_mat[:,pos];
	    a_ini[:,ii] = (U_mat[:,pos].reshape(dims[:-1],order="F"))[left:(right+1),left:(right+1)].reshape((down-up+1)*(right-left+1)*num_plane,order="F");
	    #(connect_mat_1 == pure_pix[np.where(brightness_rank == ii+1)[0]])[left:(right+1),left:(right+1)].reshape((down-up+1)*(right-left+1),order="F");
	y0 = Yt[up:(down+1),left:(right+1)].reshape((down-up+1)*(right-left+1)*num_plane,T,order="F");
	corr_img_all = np.zeros([(down-up+1)*(right-left+1)*num_plane, len(pure_pix)]);
	for ii in range(c_ini.shape[1]):
	    corr_img_all[:,ii] = vcorrcoef(y0, c_ini[:,ii]);
	corr_img_all_r = corr_img_all.reshape((down-up+1),(right-left+1)*num_plane,-1,order="F");

	return a_ini, c_ini, y0, brightness_rank, pure_pix, corr_img_all_r


def ls_solve(X, Y, mask, ind):
	"""
	least square solution.

	Parameters:
	----------------
	X: 2d np.darray 
	Y: 2d np.darray
	mask: 2d np.darray
		support constraint of coefficient beta
	ind: 2d binary np.darray
		indication matrix of whether this data is used (=1) or not (=0).

	Return:
	----------------
	beta_LS: 2d np.darray
		least square solution
	""" 

	beta_LS = np.maximum(0, np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), ind*Y));
	beta_LS = beta_LS*mask;
	return beta_LS

def make_mask(corr_img_all_r, corr, a):
	"""
	construct support constraint for spatial components.

	Parameters:
	----------------
	corr_img_all_r: 3d np.darray, d1 x d2 x number of pure superpixels
		correlation image: corr(y0, c_ini).
	corr: double scalar:
		correlation cut-off.
	a: np.darray
		initialization of spatial component.

	Return:
	----------------
	beta_LS: 2d np.darray
		least square solution
	""" 

	s = np.ones([3,3]);
	mask_a = np.zeros(a.shape);
	for ii in range(a.shape[1]):
		labeled_array, num_features = scipy.ndimage.measurements.label(corr_img_all_r[:,:,ii] > corr,structure=s);
		labeled_array = labeled_array.reshape(a.shape[0], order="F");
		temp = np.unique(labeled_array[a[:,ii]>0]); ########### which components does soma lies in ##############
		comp_pos = (np.in1d(labeled_array,temp) > 0);
		mask_a[:,ii] = comp_pos;
	return mask_a

def update_AC_l2(y0, c_ini, corr_img_all_r, corr_th_ini, corr_th_fix, corr_th_dilate, dilate_times=2, maxiter=50, tol=1e-8, whole_data=True):
	"""
	update spatial and temporal components using correlation image as constraints, with L2 loss

	Parameters:
	----------------
	y0: 2d np.darray: number pixels x T
	threshold data for this patch
	c_ini: 2d np.darray, T x number of pure superpixels
		initialization of temporal components
	corr_img_all_r: 3d np.darray, d1 x d2 x number of pure superpixels
		correlation image: corr(y0, c_ini).
	corr_th_ini: double scalar
		correlation cut-off when initializing support of spatial components.		
	corr_th_fix: double scalar
		correlation cut-off when fixing support of spatial components.		
	corr_th_dilate: list
		correlation cut-off when dilating support of spatial components.
	dilate_times: int scalar
		should be equal to the length of corr_th_dilate.
	maxiter: double scalar
		maximum iteration times
	tol: double scalar
		tolerance of change of residual
	whole_data: Boolean
		Use whole data if True or just above threshold data to do initilization.  Default is True.

	Return:
	----------------
	a_ini: 2d np.darray, number pixels x number of pure superpixels
		initialization of spatial components
	a: 2d np.darray, number pixels x number of pure superpixels
		final result of spatial components
	c_ini: 2d np.darray, T x number of pure superpixels
		initialization of temporal components
	c: 2d np.darray, T x number of pure superpixels
		final result of temporal components
	b: 2d np.darray, number pixels x 1
		constant background component
	res: list
		residual change for ||Y - AC||_F^2, should decrease monotonically.
	""" 

	############ check validation ####################
	if len(corr_th_dilate)==dilate_times - 1:
		print("correct number of correlation threshold!");
	else:
		print("wrong number of correlation threshold, use the minimum one")
		mins = min(len(corr_th_dilate), dilate_times - 1);
		corr_th_dilate = corr_th_dilate[:mins];
		dilate_times = mins;
	
	K = c_ini.shape[1];
	if whole_data:
	    ind = np.ones(y0.shape);
	else:
	    ind = (y0 > 0);
	y0 = y0*ind; ########### in case y0 doesn't satisfy sub-threshold data = 0 ##############

	a_ini = np.zeros([y0.shape[0],K]);
	c = c_ini.copy();
	f = np.ones([y0.shape[1],1]);
	g = np.ones([y0.shape[0],1]);
	corr_img_all = corr_img_all_r.reshape(y0.shape[0], -1, order="F");
	res = np.zeros(maxiter);
	
	##################### initialize A ###########################
	#mask_a = np.ones(a_ini.shape);
	mask_a = (corr_img_all > corr_th_ini);
	mask_ab = np.hstack((mask_a,g));
	temp = ls_solve(np.hstack((c,f)), y0.T, mask_ab.T, ind.T).T;
	a_ini = temp[:,:-1];
	b = temp[:,[-1]];
	#a_ini = ls_solve(c, y0.T, mask_a.T).T;
	
	mask_c = np.ones(c.shape);
	a = a_ini.copy();
	c = ls_solve(a, y0-b, mask_c.T, ind).T;
	
	##################### dilate A ##############################
	########## update mask a #################
	for ii in range(dilate_times - 1):
		mask_a = make_mask(corr_img_all_r, corr_th_dilate[ii], a);
		mask_ab = np.hstack((mask_a,g));
		temp = ls_solve(np.hstack((c,f)), y0.T, mask_ab.T, ind.T).T;
		a = temp[:,:-1];
		b = temp[:,[-1]];
		#a = ls_solve(c, y0.T, mask_a.T).T;
		c = ls_solve(a, y0-b, mask_c.T, ind).T;
		
	##################### iteratively update A and C (dilate A) ##############################
	########## update mask a #################
	mask_a = make_mask(corr_img_all_r, corr_th_fix, a);
	mask_ab = np.hstack((mask_a,g));

	for iters in range(maxiter):
		temp = ls_solve(np.hstack((c,f)), y0.T, mask_ab.T, ind.T).T;
		a = temp[:,:-1];
		b = temp[:,[-1]];
		c = ls_solve(a, y0-b, mask_c.T, ind).T;
		residual = ind*(y0 - np.matmul(a, c.T) - b);
		res[iters] = np.linalg.norm(residual, "fro");
		if iters > 0:
			if abs(res[iters] - res[iters-1])/res[iters-1] <= tol:
				break;
	
	if iters > 0:
		print(abs(res[iters] - res[iters-1])/res[iters-1]);
	return a, a_ini, c, c_ini, b, res


def fp_solve(X, Y, mask, ind, beta, mu, maxiter_nr=100, tol_nr=1):
	"""
	one-side huber loss optimization solution.

	Parameters:
	----------------
	X: 2d np.darray 
	Y: 2d np.darray
	mask: 2d np.darray
		support constraint of coefficient beta
	ind: 2d binary np.darray
		indication matrix of whether this data is used (=1) or not (=0).
	beta: 2d np.darray
		initialization of coefficient matrix beta.
	mu: double scalar
		change point of huber loss
	maxiter_nr: double scalar
		maximum iteration of newton method
	tol_nr: double scalar
		tolerance of change of coefficient

	Return:
	----------------
	beta: 2d np.darray
		one-side huber loss solution
	""" 

	X_plus = np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T);
	beta_LS = np.matmul(X_plus, ind*Y);
	if beta is None:
		beta = beta_LS*mask;
	for jj in range(maxiter_nr):
		beta0 = beta.copy();
		beta = np.maximum(0, beta_LS - np.matmul(X_plus, ind*np.maximum(0, Y - np.matmul(X, beta) - mu)));
		beta = beta*mask;
		res = np.linalg.norm(beta0 - beta, "fro");
		#print(res);
		if jj > 0:
			if res <= tol_nr:
				#print(jj);
				break
	return beta

def one_side_huber_loss(y, mu):
	"""
	objective function of one-side huber loss.

	Parameters:
	----------------
	y: 2d np.array
	mu: double scalar
		change point of huber loss

	Return:
	----------------
	objective function value
	""" 

	ind_mu = (y > mu);
	rlt = np.linalg.norm(y*(1-ind_mu), "fro")**2 + 2*sum(sum(mu*y*ind_mu)) - sum(sum(ind_mu))*(mu**2);
	return rlt

def update_AC_huber(y0, c_ini, corr_img_all_r, corr_th_ini, corr_th_fix, corr_th_dilate, dilate_times=2, mu=None, maxiter=60, tol=1e-8, maxiter_nr=100, tol_nr=1, whole_data=True):
	"""
	update spatial and temporal components using correlation image as constraints, with huber loss

	Parameters:
	----------------
	y0: 2d np.darray: number pixels x T
	threshold data for this patch
	c_ini: 2d np.darray, T x number of pure superpixels
		initialization of temporal components
	corr_img_all_r: 3d np.darray, d1 x d2 x number of pure superpixels
		correlation image: corr(y0, c_ini).
	corr_th_ini: double scalar
		correlation cut-off when initializing support of spatial components.		
	corr_th_fix: double scalar
		correlation cut-off when fixing support of spatial components.		
	corr_th_dilate: list
		correlation cut-off when dilating support of spatial components.
	dilate_times: int scalar
		should be equal to the length of corr_th_dilate.
	mu: double scalar
		change point of huber loss
	maxiter: double scalar
		maximum iteration times
	tol: double scalar
		tolerance of change of residual
	maxiter_nr: double scalar
		maximum iteration of newton method
	tol_nr: double scalar
		tolerance of change of coefficient
	whole_data: Boolean
		Use whole data if True or just above threshold data to do initilization.  Default is True.

	Return:
	----------------
	a_ini: 2d np.darray, number pixels x number of pure superpixels
		initialization of spatial components
	a: 2d np.darray, number pixels x number of pure superpixels
		final result of spatial components
	c_ini: 2d np.darray, T x number of pure superpixels
		initialization of temporal components
	c: 2d np.darray, T x number of pure superpixels
		final result of temporal components
	b: 2d np.darray, number pixels x 1
		constant background component
	res: list
		residual change for ||Y - AC||_{huber loss}^2, should decrease monotonically.
	""" 
	
	############ check validation ####################
	if len(corr_th_dilate)==dilate_times - 1:
		print("correct number of correlation threshold!");
	else:
		print("wrong number of correlation threshold, use the minimum one")
		mins = min(len(corr_th_dilate), dilate_times - 1);
		corr_th_dilate = corr_th_dilate[:mins];
		dilate_times = mins;

	K = c_ini.shape[1];
	if whole_data:
	    ind = np.ones(y0.shape);
	else:
	    ind = (y0 > 0);
	
	y0 = y0*ind; ########### in case y0 doesn't satisfy sub-threshold data = 0 ##############

	if mu is None:
		noise_std_ = noise_estimator(y0,method='logmexp');
		mu = 0.5*np.median(noise_std_);

	a_ini = np.zeros([y0.shape[0],K]);
	c = c_ini.copy();
	f = np.ones([y0.shape[1],1]);
	g = np.ones([y0.shape[0],1]);
	
	maxiter_ini = 3;
	corr_img_all = corr_img_all_r.reshape(y0.shape[0], -1, order="F");
	res = np.zeros(maxiter);
	##################### initialize A ###########################
	#mask_a = np.ones(a_ini.shape);
	mask_a = (corr_img_all > corr_th_ini);
	mask_ab = np.hstack((mask_a,g));
	temp = fp_solve(np.hstack((c,f)), y0.T, mask_ab.T, ind.T, None, mu, maxiter_nr, tol_nr).T;
	a_ini = temp[:,:-1];
	b = temp[:,[-1]];
	
	mask_c = np.ones(c.shape);
	a = a_ini.copy();    
	c = fp_solve(a, y0-b, mask_c.T, ind, c.T, mu, maxiter_nr, tol_nr).T;
	
	##################### dilate A ##############################
	########## update mask a #################
	for ii in range(dilate_times - 1):
		mask_a = make_mask(corr_img_all_r, corr_th_dilate[ii], a);
		mask_ab = np.hstack((mask_a,g));
		temp = fp_solve(np.hstack((c,f)), y0.T, mask_ab.T, ind.T, np.hstack((a,b)).T, mu, maxiter_nr, tol_nr).T;
		a = temp[:,:-1];
		b = temp[:,[-1]];

	c = fp_solve(a, y0-b, mask_c.T, ind, c.T, mu, maxiter_nr, tol_nr).T;
	
	##################### iteratively update A and C (dilate A) ##############################
	########## update mask a #################
	mask_a = make_mask(corr_img_all_r, corr_th_fix, a);
	mask_ab = np.hstack((mask_a,g));
	
	for iters in range(maxiter):
		temp = fp_solve(np.hstack((c,f)), y0.T, mask_ab.T, ind.T, np.hstack((a,b)).T, mu, maxiter_nr, tol_nr).T;
		a = temp[:,:-1];
		b = temp[:,[-1]];
	
		c = fp_solve(a, y0-b, mask_c.T, ind, c.T, mu, maxiter_nr, tol_nr).T;
		residual = ind*(y0 - np.matmul(a, c.T) - b);
		res[iters] = np.sqrt(one_side_huber_loss(residual, mu));
		if iters > 0:
			if abs(res[iters] - res[iters-1])/res[iters-1] <= tol:
				break;
	if iters > 0:
		print(abs(res[iters] - res[iters-1])/res[iters-1]);
	return a, a_ini, c, c_ini, b, res



