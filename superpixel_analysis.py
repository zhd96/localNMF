import os
import sys
import mmap
import cv2
import time

import matplotlib.pyplot as plt
import pylab
import networkx as nx
import numpy as np
import scipy.io as sio
import scipy.stats as ss
import scipy.ndimage
import scipy.signal
import scipy.sparse
import scipy
import cvxpy as cvx
import cvxopt as cvxopt

from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import NMF
from sklearn import linear_model
from scipy.ndimage.filters import convolve
from scipy.sparse import csc_matrix
from sklearn.decomposition import TruncatedSVD

def ds(data, dsize):
	dims = data.shape;
	r = int(np.floor(dims[0]/dsize[0]));
	c = int(np.floor(dims[1]/dsize[1]));
	ds_data = np.zeros([dsize[0],dsize[1],dims[2]]);
	for ii in range(dsize[0]):
		for jj in range(dsize[1]):
			if ii == dsize[0]-1:
				if jj == dsize[1]-1:
					ds_data[ii,jj,:] = data[(ii*r):,(jj*c):,:].mean(axis=0).mean(axis=0);
				else:
					ds_data[ii,jj,:] = data[(ii*r):,(jj*c):((jj+1)*c),:].mean(axis=0).mean(axis=0);
			else:
				if jj == dsize[1]-1:
					ds_data[ii,jj,:] = data[(ii*r):((ii+1)*r),(jj*c):,:].mean(axis=0).mean(axis=0);
				else:
					ds_data[ii,jj,:] = data[(ii*r):((ii+1)*r),(jj*c):((jj+1)*c),:].mean(axis=0).mean(axis=0);
	return ds_data

def resize(Y, size, interpolation=cv2.INTER_AREA):
	"""faster and 3D compatible version of skimage.transform.resize"""
	if Y.ndim == 2:
		return cv2.resize(Y, tuple(size[::-1]), interpolation=interpolation)
	elif Y.ndim == 3:
		if np.isfortran(Y):
			return (cv2.resize(np.array(
				[cv2.resize(y, size[:2], interpolation=interpolation) for y in Y.T]).T
				.reshape((-1, Y.shape[-1]), order='F'),
				(size[-1], np.prod(size[:2])), interpolation=interpolation).reshape(size, order='F'))
		else:
			return np.array([cv2.resize(y, size[:0:-1], interpolation=interpolation) for y in
					cv2.resize(Y.reshape((len(Y), -1), order='F'),
						(np.prod(Y.shape[1:]), size[0]), interpolation=interpolation)
					.reshape((size[0],) + Y.shape[1:], order='F')])
	else:  # TODO deal with ndim=4
		raise NotImplementedError

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

def mean_psd(y, method='logmexp'):
	"""
	Averaging the PSD, adapt from caiman
	Parameters:
	----------
		y: np.ndarray
		     PSD values
		method: string
			method of averaging the noise.
			Choices:
			 'mean': Mean
			 'median': Median
			 'logmexp': Exponential of the mean of the logarithm of PSD (default)
	Returns:
	-------
		mp: array
			mean psd
	"""
	if method == 'mean':
		mp = np.sqrt(np.mean(np.divide(y, 2), axis=-1))
	elif method == 'median':
		mp = np.sqrt(np.median(np.divide(y, 2), axis=-1))
	else:
		mp = np.log(np.divide((y + 1e-10), 2))
		mp = np.mean(mp, axis=-1)
		mp = np.exp(mp)
		mp = np.sqrt(mp)

	return mp

def noise_estimator(Y, noise_range=[0.25, 0.5], noise_method='logmexp', max_num_samples_fft=4000,
					opencv=True):
	"""Estimate the noise level for each pixel by averaging the power spectral density.
	Inputs:
	-------
	Y: np.ndarray
	Input movie data with time in the last axis
	noise_range: np.ndarray [2 x 1] between 0 and 0.5
	    Range of frequencies compared to Nyquist rate over which the power spectrum is averaged
	    default: [0.25,0.5]
	noise method: string
	    method of averaging the noise.
	    Choices:
	        'mean': Mean
	        'median': Median
	        'logmexp': Exponential of the mean of the logarithm of PSD (default)
	Output:
	------
	sn: np.ndarray
	    Noise level for each pixel
	"""
	T = Y.shape[-1]
	# Y=np.array(Y,dtype=np.float64)

	if T > max_num_samples_fft:
		Y = np.concatenate((Y[..., 1:max_num_samples_fft // 3 + 1],
							Y[..., np.int(T // 2 - max_num_samples_fft / 3 / 2):np.int(T // 2 + max_num_samples_fft / 3 / 2)],
							Y[..., -max_num_samples_fft // 3:]), axis=-1)
		T = np.shape(Y)[-1]

	# we create a map of what is the noise on the FFT space
	ff = np.arange(0, 0.5 + 1. / T, 1. / T)
	ind1 = ff > noise_range[0]
	ind2 = ff <= noise_range[1]
	ind = np.logical_and(ind1, ind2)
	# we compute the mean of the noise spectral density s
	if Y.ndim > 1:
		if opencv:
			import cv2
			psdx = []
			for y in Y.reshape(-1, T):
				dft = cv2.dft(y, flags=cv2.DFT_COMPLEX_OUTPUT).squeeze()[
					:len(ind)][ind]
				psdx.append(np.sum(1. / T * dft * dft, 1))
			psdx = np.reshape(psdx, Y.shape[:-1] + (-1,))
		else:
			xdft = np.fft.rfft(Y, axis=-1)
			xdft = xdft[..., ind[:xdft.shape[-1]]]
			psdx = 1. / T * abs(xdft)**2
		psdx *= 2
		sn = mean_psd(psdx, method=noise_method)

	else:
		xdft = np.fliplr(np.fft.rfft(Y))
		psdx = 1. / T * (xdft**2)
		psdx[1:] *= 2
		sn = mean_psd(psdx[ind[:psdx.shape[0]]], method=noise_method)

	return sn

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
	connect_mat_1: 2d np.darray, d1 x d2
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
	connect_mat_1 = connect_mat.reshape(Yt.shape[:-1],order='F');
	return connect_mat_1, idx-1, comps, permute_col


def spatial_temporal_ini(Yt, comps, idx, length_cut, method='svd', maxiter=5, whole_data=True, bg=False):
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
				model = NMF(n_components=1, init='custom');
				U_mat[list(comp),ii] = model.fit_transform(y_temp, W=y_temp.mean(axis=1,keepdims=True), 
											H = y_temp.mean(axis=0,keepdims=True))[:,0];
				V_mat[:,ii] = model.components_;
				ii = ii+1;
	elif method == 'iterate':
		for comp in comps:
			if(len(comp) > length_cut):
				y_temp = Yt_r[list(comp),:];
				#unique_t = np.unique(np.where(y_temp > 0)[1]);
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
	if bg:
		bg_comp_pos = np.where(U_mat.sum(axis=1) == 0)[0];
		y_temp = Yt_r[bg_comp_pos,:];
		y_temp = y_temp - y_temp.mean(axis=1,keepdims=True);
		#a, b, c = np.linalg.svd(y_temp,full_matrices=False);
		#bg_v = c[:bg,:].T;
		#bg_v = bg_v - bg_v.mean(axis=0,keepdims=True);
		svd = TruncatedSVD(n_components=bg, n_iter=7, random_state=0);
		bg_v = svd.fit(y_temp).components_.T;
		bg_v = bg_v - bg_v.mean(axis=0,keepdims=True);
	else:
		bg_v = None;

	return V_mat, U_mat, bg_v, B_mat

#def vcorrcoef(X,Y):
#	"""
#	Calculate correlation between X array and Y vector.
#
#	Parameters:
#	----------------
#	X: 2d np.darray, dimension d1 x T
#	Y: 2d np.darray, dimension d2 x T
#
#	Return:
#	----------------
#	r: 2d np.darray, dimension d1 x d2
#	correlation matrix
#	""" 
#	return np.corrcoef(X, Y)[:X.shape[0], X.shape[0]:]

def vcorrcoef(U, V, c):
	temp = (c - c.mean(axis=0,keepdims=True));
	return np.matmul(U, np.matmul(V - V.mean(axis=1,keepdims=True), temp/np.std(temp, axis=0, keepdims=True)));

def vcorrcoef2(X,y):
	Xm = np.reshape(np.mean(X,axis=1),(X.shape[0],1))
	ym = np.mean(y)
	r_num = np.sum((X-Xm)*(y-ym),axis=1)
	r_den = np.sqrt(np.sum((X-Xm)**2,axis=1)*np.sum((y-ym)**2))
	r = r_num/r_den
	return r

def search_superpixel_in_range(connect_mat, permute_col, V_mat):
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

	unique_pix = np.sort(np.unique(connect_mat))[1:];
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
		U[:,ii] = M[:,pos].copy();
		for jj in range(ii):
			U[:,ii] = U[:,ii] - U[:,jj]*sum(U[:,jj]*U[:,ii])

		U[:,ii] = U[:,ii]/np.sqrt(sum(U[:,ii]**2));
		normM = np.maximum(0, normM - np.matmul(U[:,[ii]].T, M)**2);
		normM_sqrt = np.sqrt(normM);
		ii = ii+1;
	#coef = np.matmul(np.matmul(np.linalg.inv(np.matmul(M[:,pure_pixels].T, M[:,pure_pixels])), M[:,pure_pixels].T), M);
	pure_pixels = np.array(pure_pixels);
	#coef_rank = coef.copy(); ##### from large to small 
	#for ii in range(len(pure_pixels)):
	#	coef_rank[:,ii] = [x for _,x in sorted(zip(len(pure_pixels) - ss.rankdata(coef[:,ii]), pure_pixels))];
	return pure_pixels #, coef, coef_rank

def prepare_iteration(Yd, connect_mat_1, permute_col, unique_pix, pure_pix, U_mat, V_mat, more=False, low_rank=False):
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

	dims = Yd.shape;
	T = dims[2];
	Yd = Yd.reshape(np.prod(dims[:-1]),-1, order="F");

	####################### pull out all the pure superpixels ################################
	permute_col = list(permute_col);
	pos = [permute_col.index(x) for x in pure_pix];
	U_mat = U_mat[:,pos];
	V_mat = V_mat[:,pos];
	####################### order pure superpixel according to brightness ############################
	brightness = np.zeros(len(pure_pix));

	u_max = U_mat.max(axis=0);
	v_max = V_mat.max(axis=0);
	brightness = u_max * v_max;
	brightness_arg = np.argsort(-brightness); #
	brightness_rank = U_mat.shape[1] - ss.rankdata(brightness,method="ordinal");
	U_mat = U_mat[:,brightness_arg];
	V_mat = V_mat[:,brightness_arg];
	
	temp = np.sqrt((U_mat**2).sum(axis=0,keepdims=True));
	V_mat = V_mat*temp
	U_mat = U_mat/temp;
	####################### perform svd to data #########################
	if more:
		start = time.time();
		if low_rank:
			U, S, V = np.linalg.svd(Yd, full_matrices=False);
			rank = sum(S > 1e-1);
			U = U[:,:rank]*S[:rank];
			V = V[:rank, :].T;
		else:
			U = Yd;
			V = np.identity(T);
		normalize_factor = np.std(np.matmul(U, V.T), axis=1, keepdims=True)*T;
		print(time.time()-start);
		B_mat = np.median(Yd, axis=1, keepdims=True);
		#y0 = Yt[:dims[0],:dims[1]].reshape(np.prod(dims[:-1]),T,order="F");
		#corr_img_all_r = vcorrcoef(U/normalize_factor, V.T, c_ini).reshape(dims[0],dims[1],-1,order="F");
		return U_mat, V_mat, B_mat, U, V, normalize_factor, brightness_rank, pure_pix#, corr_img_all_r
	else:
		return U_mat, V_mat, brightness_rank, pure_pix

def ls_solve_ac(X, U, V, mask=None, hals=False, beta_LS=None):
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
	#beta_LS = np.matmul(np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), U), V.T);
	if hals:
		K = X.shape[1];
		if beta_LS is None:
			beta_LS = np.zeros([K,V.shape[0]]);
		UK = np.matmul(np.matmul(X.T, U), V.T);
		VK = np.matmul(X.T, X);
		aa = np.diag(VK);
		beta_LS = beta_LS.T;
		for ii in range(K):
			#ind = np.ones((K,), bool)
			#ind[ii] = False
			#beta_LS[:,ii] = np.maximum(0, np.matmul(np.matmul(X[:,[ii]].T,np.hstack((U,-np.delete(X,ii,axis=1)))), np.hstack((V,beta_LS[:,ind])).T)/(X[:,ii]**2).sum());
			#beta_LS[:,ii] = np.maximum(0, np.matmul(np.matmul(X[:,[ii]].T,np.hstack((U,-np.delete(X,ii,axis=1)))), np.hstack((V,np.delete(beta_LS,ii,axis=1))).T)/(X[:,ii]**2).sum());
			#beta_LS[:,ii] = np.maximum(0, np.matmul(np.matmul(X[:,[ii]].T,U), V.T)/(X[:,ii]**2).sum());
			if mask is None:
				beta_LS[[ii],:] = np.maximum(0, beta_LS[[ii],:] + ((UK[[ii],:] - np.matmul(VK[[ii],:],beta_LS))/aa[ii]));
			else:
				ind = (mask[ii,:]>0);
				beta_LS[[ii],ind] = np.maximum(0, beta_LS[[ii],ind] + ((UK[[ii],ind] - np.matmul(VK[[ii],:],beta_LS[:,ind]))/aa[ii]));
		#beta_LS = beta_LS.T;
		#if mask is not None:
		#	beta_LS = beta_LS*mask;
	else:
		beta_LS = np.maximum(0, np.matmul(np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), U), V.T));
		#beta_LS = np.maximum(0, np.matmul(np.matmul(np.linalg.solve(np.matmul(X.T, X), X.T), U), V.T));
		#beta_LS = np.maximum(0, np.matmul(np.linalg.lstsq(X,U,rcond=None)[0], V.T));
		if mask is not None:
			beta_LS = beta_LS*mask;
	return beta_LS

def ls_solve_ff(X, U, V, mask):
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

	#beta_LS = np.matmul(np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), U), V.T);
	#X = csc_matrix(X);
	#X_inv = scipy.sparse.linalg.inv(csc_matrix(X.T.dot(X)));
	#temp = (X_inv.dot(X.T)).toarray();
	#beta_LS = np.maximum(0, np.matmul(np.matmul(temp, U), V.T));
	beta_LS = np.matmul(np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), U), V.T);
	#beta_LS = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), ind*Y);
	if mask is not None:
		beta_LS = beta_LS*mask;
	return beta_LS

def make_mask(corr_img_all_r, corr, mask_a, num_plane=1,times=10,max_allow_neuron_size=0.2):
	s = np.ones([3,3]);
	unit_length = int(mask_a.shape[0]/num_plane);
	dims = corr_img_all_r.shape;
	corr_img_all_r = corr_img_all_r.reshape(dims[0],int(dims[1]/num_plane),num_plane,-1,order="F");
	mask_a = mask_a.reshape(corr_img_all_r.shape,order="F");
	corr_ini = corr;
	for ii in range(mask_a.shape[-1]):
		for kk in range(num_plane):
			jj=0;
			corr = corr_ini;
			if mask_a[:,:,kk,ii].sum()>0:
				while jj<=times:
					labeled_array, num_features = scipy.ndimage.measurements.label(corr_img_all_r[:,:,kk,ii] > corr,structure=s);
					u, indices, counts = np.unique(labeled_array*mask_a[:,:,kk,ii], return_inverse=True, return_counts=True);
					#print(u);
					if len(u)==1:
						labeled_array = np.zeros(labeled_array.shape);
						if corr == 0 or corr == 1:
							break;
						else:				
							print("corr too high!")
							corr = np.maximum(0, corr - 0.1);
							jj = jj+1;
					else:
						if num_features>1:
							c = u[1:][np.argmax(counts[1:])];
							#print(c);
							labeled_array = (labeled_array==c);
							del(c);
		
						if labeled_array.sum()/unit_length < max_allow_neuron_size or corr==1 or corr==0:
							break;
						else:
							print("corr too low!")
							corr = np.minimum(1, corr + 0.1);
							jj = jj+1;
				mask_a[:,:,kk,ii] = labeled_array;
	mask_a = (mask_a*1).reshape(unit_length*num_plane,-1,order="F");
	return mask_a

def merge_components(a,c,corr_img_all_r,U,V,normalize_factor,num_list,patch_size,merge_corr_thr=0.5,merge_overlap_thr=0.8,plot_en=False):
	""" want to merge axons apatially overlap very large and temporal correlation moderately large,
	and update a and c after merge with region constrain
	Parameters:
	-----------     
	a: np.ndarray
	     matrix of spatial components (d x K)
	c: np.ndarray
	     matrix of temporal components (T x K)
	merge_corr_thr:   scalar between 0 and 1 
	    temporal correlation threshold for merging (default 0.5)
	merge_overlap_thr: scalar between 0 and 1 
	    spatial overlap threshold for merging (default 0.7)
	Returns:
	--------
	a_pri:     np.ndarray
	        matrix of merged spatial components (d x K')
	c_pri:     np.ndarray
	        matrix of merged temporal components (T x K')
	flag: merge or not
	
	"""

	f = np.ones([c.shape[0],1]);
	############ calculate overlap area ###########
	a = csc_matrix(a);
	a_corr = scipy.sparse.triu(a.T.dot(a),k=1);
	#cri = (np.corrcoef(c.T) > merge_corr_thr)*((a_corr > 0).toarray());
	cor = csc_matrix((corr_img_all_r>merge_corr_thr)*1);
	temp = cor.sum(axis=0);
	cor_corr = scipy.sparse.triu(cor.T.dot(cor),k=1);	
	cri = np.asarray((cor_corr/(temp.T)) > merge_overlap_thr)*np.asarray((cor_corr/temp) > merge_overlap_thr)*((a_corr>0).toarray());#.toarray())*(((cor_corr/(temp.T)) > merge_overlap_thr).toarray())*((a_corr > 0).toarray());
	a = a.toarray();

	#a_corr = np.matmul(a.T, a);
	#cor_corr = np.matmul(((corr_img_all_r>corr_th_fix)*1).T, (corr_img_all_r>corr_th_fix)*1);
	#cor_corr = cor_corr/(((corr_img_all_r>corr_th_fix)*1).sum(axis=0,keepdims=True));
	#cri = (cor_corr > merge_overlap_thr)*(cor_corr.T > merge_overlap_thr)*(a_corr > 0);
	#cri2 = (np.corrcoef(c.T) > merge_corr_thr)*(a_corr > 0); ########### temporally high correlated
	#
	##cri = (cri1+cri2);
	#cri = np.triu(cri, k=1);
	connect_comps = np.where(cri > 0);
	if len(connect_comps[0]) > 0:
		flag = 1;
		a_pri = a.copy();
		c_pri = c.copy();
		G = nx.Graph();
		G.add_edges_from(list(zip(connect_comps[0], connect_comps[1])))
		comps=list(nx.connected_components(G))
		merge_idx = np.unique(np.concatenate([connect_comps[0], connect_comps[1]],axis=0));
		a_pri = np.delete(a_pri, merge_idx, axis=1);
		c_pri = np.delete(c_pri, merge_idx, axis=1);
		corr_pri = np.delete(corr_img_all_r, merge_idx, axis=1);
		num_pri = np.delete(num_list,merge_idx);
		#print("merge" + str(comps));
		for comp in comps:
			comp=list(comp);
			print("merge" + str(num_list[comp]+1));
			a_zero = np.zeros([a.shape[0],1]);
			a_temp = a[:,comp];
			if plot_en:
				spatial_comp_plot(a_temp, corr_img_all_r[:,comp].reshape(patch_size[0],patch_size[1],-1,order="F"),num_list[comp],ini=False);
			mask_temp = np.where(a_temp.sum(axis=1,keepdims=True) > 0)[0];
			a_temp = a_temp[mask_temp,:];
			y_temp = np.matmul(a_temp, c[:,comp].T);
			a_temp = a_temp.mean(axis=1,keepdims=True);
			c_temp = c[:,comp].mean(axis=1,keepdims=True);
			model = NMF(n_components=1, init='custom')
			a_temp = model.fit_transform(y_temp, W=a_temp, H = (c_temp.T));

			#print("yuan" + str(np.linalg.norm(y_temp,"fro")));
			#print("jun" + str(np.linalg.norm(y_temp - np.matmul(a_temp,c_temp.T),"fro")));
			a_zero[mask_temp] = a_temp;
			c_temp = model.components_.T;
			corr_temp = vcorrcoef(U/normalize_factor, V.T, c_temp);

			a_pri = np.hstack((a_pri,a_zero));
			c_pri = np.hstack((c_pri,c_temp));
			corr_pri = np.hstack((corr_pri,corr_temp));
			num_pri = np.hstack((num_pri,num_list[comp[0]]));
		return flag, a_pri, c_pri, corr_pri, num_pri
	else:
		flag = 0;
		return flag

def delete_comp(a, c, corr_img_all_r, mask_a, num_list, temp, word, plot_en):
	print(word);
	pos = np.where(temp)[0];
	print("delete components" + str(num_list[pos]+1));
	#np.savez("delete.npz",a=a[:,pos], cor=corr_img_all_r[:,:,pos])
	if plot_en:
		spatial_comp_plot(a[:,pos], corr_img_all_r[:,:,pos], num_list=num_list[pos], ini=False);
	corr_img_all_r = np.delete(corr_img_all_r, pos, axis=2);
	mask_a = np.delete(mask_a, pos, axis=1);
	a = np.delete(a, pos, axis=1);
	c = np.delete(c, pos, axis=1);
	num_list = np.delete(num_list, pos);
	return a, c, corr_img_all_r, mask_a, num_list


def update_AC_l2(U, V, normalize_factor, a, c, b, patch_size, corr_th_fix, 
			maxiter=50, tol=1e-8, update_after=None,merge_corr_thr=0.5,merge_overlap_thr=0.7, num_plane=1, plot_en=False,hals=False, max_allow_neuron_size=0.2):

	K = c.shape[1];
	res = np.zeros(maxiter);
	uv_mean = (U*(V.mean(axis=0,keepdims=True))).sum(axis=1,keepdims=True);

	## initialize spatial support ##
	mask_a = (a>0)*1;
	corr_img_all = vcorrcoef(U/normalize_factor, V.T, c);
	corr_img_all_r = corr_img_all.reshape(patch_size[0],patch_size[1],-1,order="F");

	## initialize bg ##
	f = np.ones([c.shape[0],1]);
	num_list = np.arange(K);

	for iters in range(maxiter):
		start = time.time();
		a = ls_solve_ac(c, np.hstack((V,-1*f)), np.hstack((U,b)), mask=mask_a.T, hals=hals, beta_LS=a).T;
		temp = (a.sum(axis=0) == 0);
		if sum(temp):
			a, c, corr_img_all_r, mask_a, num_list = delete_comp(a, c, corr_img_all_r, mask_a, num_list, temp, "zero a!", plot_en);		
		b = np.maximum(0, uv_mean-((a*(c.mean(axis=0,keepdims=True))).sum(axis=1,keepdims=True)));

		c = ls_solve_ac(a, np.hstack((U,b)), np.hstack((V,-1*f)), mask=None, hals=hals, beta_LS=c).T;
		temp = (c.sum(axis=0) == 0);
		if sum(temp):
			a, c, corr_img_all_r, mask_a, num_list = delete_comp(a, c, corr_img_all_r, mask_a, num_list, temp, "zero c!", plot_en);
		b = np.maximum(0, uv_mean-(a*(c.mean(axis=0,keepdims=True))).sum(axis=1,keepdims=True));

		if update_after and ((iters + 1) % update_after == 0):
			corr_img_all = vcorrcoef(U/normalize_factor, V.T, c);
			rlt = merge_components(a,c,corr_img_all,U, V, normalize_factor,num_list,patch_size,merge_corr_thr=merge_corr_thr,merge_overlap_thr=merge_overlap_thr,plot_en=plot_en);
			flag = isinstance(rlt, int);
			if ~np.array(flag):
				a = rlt[1];
				c = rlt[2];
				corr_img_all = rlt[3];#vcorrcoef(U/normalize_factor, V.T, c);
				num_list = rlt[4];
			else:
				print("no merge!");
			mask_a = (a>0)*1;
			corr_img_all_r = corr_img_all.reshape(patch_size[0],patch_size[1],-1,order="F");
			mask_a = make_mask(corr_img_all_r, corr_th_fix, mask_a, num_plane, max_allow_neuron_size=max_allow_neuron_size);

			temp = (mask_a.sum(axis=0) == 0);
			if sum(temp):
				a, c, corr_img_all_r, mask_a, num_list = delete_comp(a, c, corr_img_all_r, mask_a, num_list, temp, "zero mask!", plot_en);
			a = a*mask_a;

		residual = (np.matmul(U, V.T) - np.matmul(a, c.T) - b);
		res[iters] = np.linalg.norm(residual, "fro");
		print(res[iters]);
		print("time: " + str(time.time()-start));
		if iters > 0:
			if abs(res[iters] - res[iters-1])/res[iters-1] <= tol:
				break;
	temp = np.sqrt((a**2).sum(axis=0,keepdims=True));
	c = c*temp;
	a = a/temp;
	brightness = np.zeros(a.shape[1]);
	a_max = a.max(axis=0);
	c_max = c.max(axis=0);
	brightness = a_max * c_max;
	brightness_rank = np.argsort(-brightness); #a.shape[1] - ss.rankdata(brightness,method="ordinal");
	a = a[:,brightness_rank];
	c = c[:,brightness_rank];
	corr_img_all_r = corr_img_all_r[:,:,brightness_rank];
	num_list = num_list[brightness_rank];
	ff = None;
	fb = None;
	if iters > 0:
		print("residual relative change: " + str(abs(res[iters] - res[iters-1])/res[iters-1]));
	return a, c, b, fb, ff, res, corr_img_all_r, num_list


def update_AC_bg_l2(U, V, normalize_factor, a, c, b, patch_size, corr_th_fix, 
			maxiter=50, tol=1e-8, update_after=None,merge_corr_thr=0.5,merge_overlap_thr=0.7, num_plane=1, plot_en=False, ff=None,hals=False,max_allow_neuron_size=0.2):
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
	
	K = c.shape[1];
	num_bg = 1+ff.shape[1];

	f = np.ones([c.shape[0],1]);
	g = np.ones([a.shape[0],num_bg]);
	res = np.zeros(maxiter);
	
	###################### initialize A ###########################
	mask_a = (a > 0)*1;
	#mask_a = (corr_img_all > corr_th_ini);
	mask_ab = np.hstack((mask_a,g));
	temp = ls_solve_ac(np.hstack((c,f,ff)), V, U, mask_ab.T).T;

	a = temp[:,:-num_bg];
	fb = temp[:,(-num_bg+1):];
	#b = temp[:,[-1]];
	#a = ls_solve_a(c, V, U, mask_a.T).T;
	b = np.maximum(0, (U*(V.mean(axis=0,keepdims=True))).sum(axis=1,keepdims=True)-(a*(c.mean(axis=0,keepdims=True))).sum(axis=1,keepdims=True));
	num_list = np.arange(K);

	try:
		c = ls_solve_ac(a, np.hstack((U,b,fb)), np.hstack((V,-1*f,-1*ff)), None).T;	
		#c = c-c.min(axis=0,keepdims=True);
		b = (U*(V.mean(axis=0,keepdims=True))).sum(axis=1,keepdims=True)-(a*(c.mean(axis=0,keepdims=True))).sum(axis=1,keepdims=True);
		ff = ls_solve_ff(fb, np.hstack((U,b,a)), np.hstack((V,-1*f,-1*c)), None).T;
		b = np.maximum(0, b);

	except:
		print("zero a!");
		pos = np.where(a.sum(axis=0) == 0)[0];
		print("delete components" + str(num_list[pos]+1));
		a = np.delete(a, pos, axis=1);
		num_list = np.delete(num_list, pos);
		c = ls_solve_ac(a, np.hstack((U,b,fb)), np.hstack((V,-1*f,-1*ff)), None).T;	
		#c = c-c.min(axis=0,keepdims=True);
		b = (U*(V.mean(axis=0,keepdims=True))).sum(axis=1,keepdims=True)-(a*(c.mean(axis=0,keepdims=True))).sum(axis=1,keepdims=True);
		ff = ls_solve_ff(fb, np.hstack((U,b,a)), np.hstack((V,-1*f,-1*c)), None).T;
		b = np.maximum(0, b);
			
	##################### iteratively update A and C (dilate A) ##############################
	########## update mask a #################
	print("dilate!");
	mask_a = (a > 0)*1;
	corr_img_all_r = vcorrcoef(U/normalize_factor, V.T, c).reshape(patch_size[0],patch_size[1],-1,order="F");
	mask_a = make_mask(corr_img_all_r, corr_th_fix, mask_a, num_plane,max_allow_neuron_size=max_allow_neuron_size);
	a = a*mask_a;

	if sum(mask_a.sum(axis=0) == 0):
		print("zero mask a!");
		pos = np.where(mask_a.sum(axis=0) == 0)[0];
		print("delete components" + str(num_list[pos]+1));
		#np.savez("delete.npz",a=a[:,pos], cor=corr_img_all_r[:,:,pos])
		if plot_en:
			spatial_comp_plot(a[:,pos], corr_img_all_r[:,:,pos], num_list=num_list[pos], ini=False);
		corr_img_all_r = np.delete(corr_img_all_r, pos, axis=2);
		mask_a = np.delete(mask_a, pos, axis=1);
		a = np.delete(a, pos, axis=1);
		c = np.delete(c, pos, axis=1);
		num_list = np.delete(num_list, pos);
	mask_ab = np.hstack((mask_a,g));

	for iters in range(maxiter):
		start_time = time.time();
		try:
			temp = ls_solve_ac(np.hstack((c,f,ff)), V, U, mask_ab.T, hals, np.hstack((a,b,fb))).T;
			a = temp[:,:-num_bg];
			fb = temp[:,(-num_bg+1):];
			#a = temp[:,:-1];
			#b = temp[:,[-1]];
			#a = ls_solve_a(c, V, U, mask_a.T).T;
			b = np.maximum(0, (U*(V.mean(axis=0,keepdims=True))).sum(axis=1,keepdims=True)-(a*(c.mean(axis=0,keepdims=True))).sum(axis=1,keepdims=True));
			#residual = (np.matmul(U, V.T) - np.matmul(a, c.T) - b);
			#res[iters] = np.linalg.norm(residual, "fro");
			#print(res[iters]);
		except:
			print("zero c!");
			pos = np.where(c.sum(axis=0) == 0)[0];
			print("delete components" + str(num_list[pos]+1));
			if plot_en:
				spatial_comp_plot(a[:,pos], corr_img_all_r[:,:,pos], num_list[pos], ini=False);
			mask_a = np.delete(mask_a, pos, axis=1);
			mask_ab = np.hstack((mask_a,g));
			corr_img_all_r = np.delete(corr_img_all_r, pos, axis=2);
			c = np.delete(c, pos, axis=1);
			num_list = np.delete(num_list, pos);
			temp = ls_solve_ac(np.hstack((c,f,ff)), V, U, mask_ab.T, hals, np.hstack((a,b,fb))).T;
			a = temp[:,:-num_bg];
			fb = temp[:,(-num_bg+1):];
			#a = temp[:,:-1];
			#b = temp[:,[-1]];
			#a = ls_solve_a(c, V, U, mask_a.T).T;
			b = np.maximum(0, (U*(V.mean(axis=0,keepdims=True))).sum(axis=1,keepdims=True)-(a*(c.mean(axis=0,keepdims=True))).sum(axis=1,keepdims=True));

			#residual = (np.matmul(U, V.T) - np.matmul(a, c.T) - b);
			#res[iters] = np.linalg.norm(residual, "fro");
			#print(res[iters]);	
		try:
			c = ls_solve_ac(a, np.hstack((U,b,fb)), np.hstack((V,-1*f,-1*ff)), None, hals, c).T;	
			#c = c-c.min(axis=0,keepdims=True);
			b = (U*(V.mean(axis=0,keepdims=True))).sum(axis=1,keepdims=True)-(a*(c.mean(axis=0,keepdims=True))).sum(axis=1,keepdims=True);
			ff = ls_solve_ff(fb, np.hstack((U,b,a)), np.hstack((V,-1*f,-1*c)), None).T;
			b = np.maximum(0, b);
	
		except:
			print("zero a!");
			pos = np.where(a.sum(axis=0) == 0)[0];
			print("delete components" + str(num_list[pos]+1));
			a = np.delete(a, pos, axis=1);
			num_list = np.delete(num_list, pos);
			c = ls_solve_ac(a, np.hstack((U,b,fb)), np.hstack((V,-1*f,-1*ff)), None, hals, c).T;	
			#c = c-c.min(axis=0,keepdims=True);
			b = (U*(V.mean(axis=0,keepdims=True))).sum(axis=1,keepdims=True)-(a*(c.mean(axis=0,keepdims=True))).sum(axis=1,keepdims=True);
			ff = ls_solve_ff(fb, np.hstack((U,b,a)), np.hstack((V,-1*f,-1*c)), None).T;
			b = np.maximum(0, b);
		# Merge Components
		if update_after and ((iters + 1) % update_after == 0):
			corr_img_all = vcorrcoef(U/normalize_factor, V.T, c);
			rlt = merge_components(a,c,corr_img_all,U, V, normalize_factor,num_list,patch_size,merge_corr_thr=merge_corr_thr,merge_overlap_thr=merge_overlap_thr,plot_en=plot_en);
			flag = isinstance(rlt, int);
			if ~np.array(flag):
				a = rlt[1];
				c = rlt[2];
				corr_img_all = rlt[3];#vcorrcoef(U/normalize_factor, V.T, c);
				num_list = rlt[4];
			else:
				print("no merge!");
			mask_a = (a>0)*1;
			corr_img_all_r = corr_img_all.reshape(patch_size[0],patch_size[1],-1,order="F");
			mask_a = make_mask(corr_img_all_r, corr_th_fix, mask_a, num_plane, max_allow_neuron_size=max_allow_neuron_size);

			a_temp = (mask_a.sum(axis=0) == 0);
			#corr_temp = (corr_img_all_r.max(axis=0).max(axis=0) < corr_good);
			if sum(a_temp): #or sum(corr_temp):
				#if sum(a_temp):
				print("zero mask a!")
				pos = np.where(a_temp)[0];
				#if sum(corr_temp):
				#	print("junk components!");
				#	pos = np.where(corr_temp)[0];
				print("delete components" + str(num_list[pos]+1));
				#np.savez("delete.npz",a=a[:,pos], cor=corr_img_all_r[:,:,pos])
				if plot_en:
					spatial_comp_plot(a[:,pos], corr_img_all_r[:,:,pos], num_list[pos], ini=False);
				corr_img_all_r = np.delete(corr_img_all_r, pos, axis=2);
				mask_a = np.delete(mask_a, pos, axis=1);
				a = np.delete(a, pos, axis=1);
				c = np.delete(c, pos, axis=1);
				num_list = np.delete(num_list, pos);
			mask_ab = np.hstack((mask_a,g));
			a = a*mask_a;

		residual = (np.matmul(U, V.T) - np.matmul(a, c.T) - b - np.matmul(fb,ff.T));
		res[iters] = np.linalg.norm(residual, "fro");
		print(res[iters]);
		print("time: " + str(time.time()-start_time));

		if iters > 0:
			if abs(res[iters] - res[iters-1])/res[iters-1] <= tol:
				break;
	temp = np.sqrt((a**2).sum(axis=0,keepdims=True));
	c = c*temp;
	a = a/temp;
	brightness = np.zeros(a.shape[1]);
	a_max = a.max(axis=0);
	c_max = c.max(axis=0);
	brightness = a_max * c_max;
	brightness_rank = np.argsort(-brightness); #a.shape[1] - ss.rankdata(brightness,method="ordinal");
	a = a[:,brightness_rank];
	c = c[:,brightness_rank];
	corr_img_all_r = corr_img_all_r[:,:,brightness_rank];
	num_list = num_list[brightness_rank];

	#for ii in range(a.shape[1]):
	#	v_max = a[:,ii].max();
	#	u_max = c[:,ii].max();
	#	brightness[ii] = u_max * v_max;
	#brightness_rank = a.shape[1] - ss.rankdata(brightness,method="ordinal");
	#brightness_rank = np.arange(a.shape[1]);
	#a_cp = a.copy();
	#c_cp = c.copy();
	#corr_cp = corr_img_all_r.copy();
	#num_list_cp = num_list.copy();
	#for ii in range(a.shape[1]):
	#	a_cp[:,ii] = a[:,np.where(brightness_rank==ii)[0][0]];
	#	c_cp[:,ii] = c[:,np.where(brightness_rank==ii)[0][0]];
	#	corr_cp[:,:,ii] = corr_img_all_r[:,:,np.where(brightness_rank==ii)[0][0]];
	#	num_list_cp[ii] = num_list[np.where(brightness_rank==ii)[0][0]];
	if iters > 0:
		print("residual relative change: " + str(abs(res[iters] - res[iters-1])/res[iters-1]));
	return a, c, b, fb, ff, res, corr_img_all_r, num_list

def reconstruct(Yd, spatial_components, temporal_components, background_components, fb=None, ff=None):
	"""
	generate reconstruct movie, and get residual
	Parameters:
	---------------
	Yd: np.darray: d1 x d2 x T
	spatial_components: np.darray: d x K
	temporal_components: np.darray: T x K

	"""
	#up = x_range[0];
	#down = x_range[1];
	#left = y_range[0];
	#right = y_range[1];

	y0 = Yd#[up:down, left:right, :];
	dims = y0.shape;
	if fb is not None:
		mov_res = y0 - (np.matmul(spatial_components, temporal_components.T)+np.matmul(fb, ff.T)+background_components).reshape(dims, order='F');
	else:
		mov_res = y0 - (np.matmul(spatial_components, temporal_components.T)+background_components).reshape(dims, order='F');
	return mov_res

def superpixel_initialization(Yd, cut_off_point=0.9, length_cut=10, th=2, num_plane=1, plot_en=False, text=False):

	## make sure Yd is positive ##
	dims = Yd.shape[:2];
	T = Yd.shape[2];

	Yt = threshold_data(Yd, th=th);
	
	if num_plane > 1:
		print("3d data!");
		connect_mat_1, idx, comps, permute_col = find_superpixel_3d(Yt,num_plane,cut_off_point,length_cut,eight_neighbours=True);
	else:
		start = time.time();
		print("find superpixels!")
		connect_mat_1, idx, comps, permute_col = find_superpixel(Yt,cut_off_point,length_cut,eight_neighbours=True);
		print(time.time()-start);
	start = time.time();
	print("rank 1 svd!")
	c_ini, a_ini, _ , _ = spatial_temporal_ini(Yt, comps, idx, length_cut, maxiter=5, whole_data=True, method="svd");
	print(time.time()-start);
		
	temp = np.sqrt((a_ini**2).sum(axis=0,keepdims=True));
	c_ini = c_ini*temp;
	a_ini = a_ini/temp;
	brightness = np.zeros(a_ini.shape[1]);
	a_max = a_ini.max(axis=0);
	c_max = c_ini.max(axis=0);
	brightness = a_max * c_max;
	brightness_arg = np.argsort(-brightness); #
	a_ini = a_ini[:,brightness_arg];
	c_ini = c_ini[:,brightness_arg];
	brightness_rank = a_ini.shape[1] - ss.rankdata(brightness,method="ordinal");
	if plot_en:
		Cnt = local_correlations_fft(Yt);
		superpixel_corr_plot(connect_mat_1, permute_col, Cnt, brightness_rank, text);
	else:
		Cnt = None;
	return a_ini, c_ini, connect_mat_1, brightness_rank, permute_col, Cnt


def axon_pipeline(Yd, cut_off_point=[0.9,0.8], length_cut=[15,10], th=[2,1], pass_num=2, residual_cut = 0.43,
					corr_th_fix=0.4, max_allow_neuron_size=0.3, merge_corr_thr=0.5, merge_overlap_thr=0.8, num_plane=1, patch_size=[100,100],
					plot_en=False, TF=False, fudge_factor=1, text=True, bg=False, max_iter=21, max_iter_fin=41, 
					hals=False, low_rank=False, update_after=2):

	dims = Yd.shape[:2];
	T = Yd.shape[2];
	superpixel_rlt = [];
	## cut image into small parts to find pure superpixels ##
	
	patch_height = patch_size[0];#100;
	patch_width = patch_size[1];#100;
	height_num = int(np.ceil(dims[0]/patch_height));  ########### if need less data to find pure superpixel, change dims[0] here #################
	width_num = int(np.ceil(dims[1]/(patch_width*num_plane)));
	num_patch = height_num*width_num;
	patch_ref_mat = np.array(range(num_patch)).reshape(height_num, width_num, order="F");

	ii = 0;
	while ii < pass_num:
		print("start " + str(ii+1) + " pass!");
		if ii > 0:
			if bg:
				Yd_res = reconstruct(Yd, a, c, b, fb, ff);
			else:
				Yd_res = reconstruct(Yd, a, c, b);
			Yt = threshold_data(Yd_res, th=th[ii]);
		else:
			if th[ii] >= 0:
				Yt = threshold_data(Yd, th=th[ii]);
			else:
				Yt = Yd.copy();
		
		start = time.time();
		if num_plane > 1:
			print("3d data!");
			connect_mat_1, idx, comps, permute_col = find_superpixel_3d(Yt,num_plane,cut_off_point[ii],length_cut[ii],eight_neighbours=True);
		else:
			print("find superpixels!")
			connect_mat_1, idx, comps, permute_col = find_superpixel(Yt,cut_off_point[ii],length_cut[ii],eight_neighbours=True);
		print(time.time()-start);
		
		start = time.time();
		print("rank 1 svd!")
		if ii > 0:
			c_ini, a_ini, _ , _ = spatial_temporal_ini(Yt, comps, idx, length_cut[ii], maxiter=5, whole_data=True, method="svd",bg=bg);
		else:
			c_ini, a_ini, ff, _ = spatial_temporal_ini(Yt, comps, idx, length_cut[ii], maxiter=5, whole_data=True, method="svd",bg=bg);
			#return ff
		print(time.time()-start);
		unique_pix = np.asarray(np.sort(np.unique(connect_mat_1))[1:]); #search_superpixel_in_range(connect_mat_1, permute_col, V_mat);
		#unique_pix = np.asarray(unique_pix);
		pure_pix = [];
		
		start = time.time();
		print("find pure superpixels!")
		for kk in range(num_patch):
			pos = np.where(patch_ref_mat==kk);
			up=pos[0][0]*patch_height;
			down=min(up+patch_height, dims[0]);
			left=pos[1][0]*patch_width;
			right=min(left+patch_width, dims[1]);
			unique_pix_temp, M = search_superpixel_in_range((connect_mat_1.reshape(dims[0],int(dims[1]/num_plane),num_plane,order="F"))[up:down,left:right], permute_col, c_ini);
			pure_pix_temp = fast_sep_nmf(M, M.shape[1], residual_cut[ii]);
			if len(pure_pix_temp)>0:
				pure_pix = np.hstack((pure_pix, unique_pix_temp[pure_pix_temp]));

		pure_pix = np.unique(pure_pix);
		print(time.time()-start);
		if plot_en:
			pure_superpixel_compare_plot(connect_mat_1, unique_pix, pure_pix, text);
		
		start = time.time();		
		print("prepare iteration!")	
		if ii > 0:
			a_ini, c_ini, brightness_rank, pure_pix = prepare_iteration(Yd_res, connect_mat_1, permute_col, unique_pix, pure_pix, a_ini, c_ini);
			a = np.hstack((a, a_ini));
			c = np.hstack((c, c_ini));
		else:
			a, c, b, U, V, normalize_factor, brightness_rank, pure_pix = prepare_iteration(Yd, connect_mat_1, permute_col, unique_pix, pure_pix, a_ini, c_ini, more=True, low_rank=low_rank);
		
		print(time.time()-start);
		
		if plot_en:
			Cnt = local_correlations_fft(Yt);
			superpixel_corr_plot(connect_mat_1, pure_pix, Cnt, brightness_rank, text);
			#temporal_comp_plot(c_ini,ini = True);
			#spatial_comp_plot(a_ini, corr_img_all_r, ini=True);
		print("start " + str(ii+1) + " pass iteration!")
		if ii == pass_num - 1:
			maxiter = max_iter_fin;
		else:
			maxiter=max_iter;
		
		if bg:
			a, c, b, fb, ff, res, corr_img_all_r, num_list = update_AC_bg_l2(U, V, normalize_factor, a, c, b, dims,
										corr_th_fix, maxiter=maxiter, tol=1e-8, update_after=update_after,
										merge_corr_thr=merge_corr_thr,merge_overlap_thr=merge_overlap_thr, num_plane=num_plane, plot_en=plot_en, ff=ff,hals=hals,max_allow_neuron_size=max_allow_neuron_size);

		else:
			a, c, b, fb, ff, res, corr_img_all_r, num_list = update_AC_l2(U, V, normalize_factor, a, c, b, dims,
										corr_th_fix, maxiter=maxiter, tol=1e-8, update_after=update_after,
										merge_corr_thr=merge_corr_thr,merge_overlap_thr=merge_overlap_thr, num_plane=num_plane, plot_en=plot_en,hals=hals,max_allow_neuron_size=max_allow_neuron_size);
		#if plot_en:
		#	temporal_comp_plot(c,num_list,ini = False);
			#spatial_comp_plot(a, corr_img_all_r, ini=False);
		superpixel_rlt.append({'connect_mat_1':connect_mat_1, 'pure_pix':pure_pix, 'brightness_rank':brightness_rank});
		if pass_num > 1 and ii == 0:
			rlt = {'a':a, 'c':c, 'b':b, "fb":fb, "ff":ff, 'res':res, 'corr_img_all_r':corr_img_all_r, 'num_list':num_list};
			#np.save("full_1pass.npy",rlt);
			a0 = a.copy();
		ii = ii+1;

	c_tf = [];
	if TF:
		sigma = noise_estimator(c.T);
		sigma *= fudge_factor
		for ii in range(c.shape[1]):
			c_tf = np.hstack((c_tf, l1_tf(c[:,ii], sigma[ii])));
		c_tf = c_tf.reshape(T,int(c_tf.shape[0]/T),order="F");
	#	if plot_en:
	#		temporal_compare_plot(c, c_tf, ini = False);
	#else:
	#	if plot_en:
	#		temporal_comp_plot(c, num_list, ini = False);

	if plot_en:
		#spatial_comp_plot(a_fin, corr_img_all_r_fin, ini=False);
		if pass_num > 1:
			spatial_sum_plot(a0, a, dims, num_list, text);
		Yd_res = reconstruct(Yd, a, c, b);
		Yd_res = threshold_data(Yd_res, th=0);
		Cnt = local_correlations_fft(Yd_res);
		scale = np.maximum(1, int(Cnt.shape[1]/Cnt.shape[0]));
		plt.figure(figsize=(8*scale,8))
		ax1 = plt.subplot(1,1,1);
		show_img(ax1, Cnt);
		ax1.set(title="Local mean correlation for residual")
		ax1.title.set_fontsize(15)
		ax1.title.set_fontweight("bold")
		plt.show();
	fin_rlt = {'a':a, 'c':c, 'c_tf':c_tf, 'b':b, "fb":fb, "ff":ff, 'res':res, 'corr_img_all_r':corr_img_all_r, 'num_list':num_list};
	if pass_num > 1:
		return {'rlt':rlt, 'fin_rlt':fin_rlt, "superpixel_rlt":superpixel_rlt}
	else:
		return {'fin_rlt':fin_rlt, "superpixel_rlt":superpixel_rlt}

def l1_tf(y, sigma):
	if np.abs(sigma/y.max())<=1e-3:
		print('Do not denoise (high SNR: noise_level=%.3e)'%sigma);
		return y
#
	n = y.size
	# Form second difference matrix.
	D = (np.diag(2*np.ones(n),0)+np.diag(-1*np.ones(n-1),1)+np.diag(-1*np.ones(n-1),-1))[1:n-1];
	x = cvx.Variable(n)
	obj = cvx.Minimize(cvx.norm(D*x, 1));
	constraints = [cvx.norm(y-x,2)<=sigma*np.sqrt(n)]
	prob = cvx.Problem(obj, constraints)    
#
	prob.solve(solver=cvx.ECOS,verbose=False)
	
	# Check for error.
	if prob.status != cvx.OPTIMAL: 
		raise Exception("Solver did not converge!")
		return y
	return np.asarray(x.value).flatten()


def match_comp(rlt, rlt_lasso_Ydc, rlt_lasso_Yrawc, rlt_a, rlt_lasso_Yda, rlt_lasso_Yrawa,th):
	K = rlt.shape[1];
	order_Yd = np.zeros([K])
	order_Yraw = np.zeros([K])
	for ii in range(K):
		temp = vcorrcoef2(rlt_lasso_Ydc.T, rlt[:,ii]);
		temp2 = vcorrcoef2(rlt_lasso_Yrawc.T, rlt[:,ii]);
		pos = np.argsort(-temp)[:sum(temp > th)];
		pos2 = np.argsort(-temp2)[:sum(temp2 > th)];

		if len(pos)>0:
			spa_temp = np.where(np.matmul(rlt_a[:,[ii]].T, rlt_lasso_Yda[:,pos])>0)[1];
			if len(spa_temp)>0:
				order_Yd[ii] = int(pos[spa_temp[0]]);
			else:
				order_Yd[ii] = np.nan;
		else:
			order_Yd[ii] = np.nan;

		if len(pos2)>0:
			spa_temp2 = np.where(np.matmul(rlt_a[:,[ii]].T, rlt_lasso_Yrawa[:,pos2])>0)[1];
			if len(spa_temp2)>0:
				order_Yraw[ii] = int(pos2[spa_temp2[0]]);
			else:
				order_Yraw[ii] = np.nan;
		else:
			order_Yraw[ii] = np.nan;
	order_Yd = np.asarray(order_Yd,dtype=int);
	order_Yraw = np.asarray(order_Yraw,dtype=int);
	return order_Yd, order_Yraw


def match_comp_gt(rlt_gt, rlt, rlt_lasso_Ydc, rlt_lasso_Yrawc,rlt_gta, rlt_a, rlt_lasso_Yda, rlt_lasso_Yrawa,th):
	K = rlt_gt.shape[1];
	order_Ys = np.zeros([K]);
	order_Yd = np.zeros([K])
	order_Yraw = np.zeros([K])
	for ii in range(K):
		temp0 = vcorrcoef2(rlt.T, rlt_gt[:,ii]);
		temp = vcorrcoef2(rlt_lasso_Ydc.T, rlt_gt[:,ii]);
		temp2 = vcorrcoef2(rlt_lasso_Yrawc.T, rlt_gt[:,ii]);
		pos0 = np.argsort(-temp0)[:sum(temp0 > th)];
		pos = np.argsort(-temp)[:sum(temp > th)];
		pos2 = np.argsort(-temp2)[:sum(temp2 > th)];

		if len(pos0)>0:
			spa_temp0 = np.where(np.matmul(rlt_gta[:,[ii]].T, rlt_a[:,pos0])>0)[1];
			if len(spa_temp0)>0:
				#print(int(pos0[spa_temp0]));
				order_Ys[ii] = int(pos0[spa_temp0[0]]);
			else:
				order_Ys[ii] = np.nan;
			#if ii == K-1:
			#	order_Ys[ii] = 13;
		else:
			order_Ys[ii] = np.nan;

		if len(pos)>0:
			spa_temp = np.where(np.matmul(rlt_gta[:,[ii]].T, rlt_lasso_Yda[:,pos])>0)[1];
			if len(spa_temp)>0:
				order_Yd[ii] = int(pos[spa_temp[0]]);
			else:
				order_Yd[ii] = np.nan;
		else:
			order_Yd[ii] = np.nan;

		if len(pos2)>0:
			spa_temp2 = np.where(np.matmul(rlt_gta[:,[ii]].T, rlt_lasso_Yrawa[:,pos2])>0)[1];
			if len(spa_temp2)>0:
				order_Yraw[ii] = int(pos2[spa_temp2[0]]);
			else:
				order_Yraw[ii] = np.nan;
		else:
			order_Yraw[ii] = np.nan;
	order_Ys = np.asarray(order_Ys,dtype=int);
	order_Yd = np.asarray(order_Yd,dtype=int);
	order_Yraw = np.asarray(order_Yraw,dtype=int);
	return order_Ys, order_Yd, order_Yraw

def match_comp_projection(rlt_xyc, rlt_yzc, rlt_xya, rlt_yza, dims1, dims2, th):
	K = rlt_xyc.shape[1];
	order = np.zeros([K]);
	rlt_xya = rlt_xya.reshape(dims1[0],dims1[1],-1,order="F");
	rlt_yza = rlt_yza.reshape(dims2[0],dims2[1],-1,order="F");

	for ii in range(K):
		temp0 = vcorrcoef2(rlt_yzc.T, rlt_xyc[:,ii]);
		pos0 = np.argsort(-temp0)[:sum(temp0 > th)];

		if len(pos0)>0:
			spa_temp0 = np.where(np.matmul(rlt_xya[:,:,[ii]].sum(axis=0).T, rlt_yza[:,:,pos0].sum(axis=0))>0)[1];
			#print(spa_temp0);
			if len(spa_temp0)>0:
				#print(int(pos0[spa_temp0]));
				order[ii] = int(pos0[spa_temp0[0]]);
			else:
				order[ii] = np.nan;
		else:
			order[ii] = np.nan;
	order = np.asarray(order,dtype=int);
	return order

def clip_data(rlt_a, rlt_lasso_Yda, rlt_lasso_Yrawa,th):
	a_out = np.percentile(rlt_a,th,axis=0);
	a_lasso_den_out = np.percentile(rlt_lasso_Yda,th,axis=0);
	a_lasso_raw_out = np.percentile(rlt_lasso_Yrawa,th,axis=0);
	a = rlt_a.copy();
	a_lasso_den = rlt_lasso_Yda.copy();
	a_lasso_raw = rlt_lasso_Yrawa.copy();
	for ii in range(a.shape[1]):
		if a_out[ii] > 0:
			a[:,ii] = np.clip(rlt_a[:,ii],0,a_out[ii]);
	for ii in range(a_lasso_den.shape[1]):
		if a_lasso_den_out[ii] >0:
			a_lasso_den[:,ii] = np.clip(rlt_lasso_Yda[:,ii],0,a_lasso_den_out[ii]);
	for ii in range(a_lasso_raw.shape[1]):
		if a_lasso_raw_out[ii] >0:
			a_lasso_raw[:,ii] = np.clip(rlt_lasso_Yrawa[:,ii],0,a_lasso_raw_out[ii]);
	return a, a_lasso_den, a_lasso_raw

def clip_data_gt(rlt_gta, rlt_a, rlt_lasso_Yda, rlt_lasso_Yrawa,th):
	a_gt_out = np.percentile(rlt_gta,th,axis=0);
	a_out = np.percentile(rlt_a,th,axis=0);
	a_lasso_den_out = np.percentile(rlt_lasso_Yda,th,axis=0);
	a_lasso_raw_out = np.percentile(rlt_lasso_Yrawa,th,axis=0);
	a_gt = rlt_gta.copy();
	a = rlt_a.copy();
	a_lasso_den = rlt_lasso_Yda.copy();
	a_lasso_raw = rlt_lasso_Yrawa.copy();
	for ii in range(a_gt.shape[1]):
		if a_gt_out[ii] > 0:
			a_gt[:,ii] = np.clip(rlt_gta[:,ii],0,a_gt_out[ii]);	
	for ii in range(a.shape[1]):
		if a_out[ii] > 0:
			a[:,ii] = np.clip(rlt_a[:,ii],0,a_out[ii]);
	for ii in range(a_lasso_den.shape[1]):
		if a_lasso_den_out[ii] >0:
			a_lasso_den[:,ii] = np.clip(rlt_lasso_Yda[:,ii],0,a_lasso_den_out[ii]);
	for ii in range(a_lasso_raw.shape[1]):
		if a_lasso_raw_out[ii] >0:
			a_lasso_raw[:,ii] = np.clip(rlt_lasso_Yrawa[:,ii],0,a_lasso_raw_out[ii]);
	return a_gt, a, a_lasso_den, a_lasso_raw

def pure_superpixel_compare_plot(connect_mat_1, unique_pix, pure_pix, text=False):
	scale = np.maximum(1, (connect_mat_1.shape[1]/connect_mat_1.shape[0]));
	fig = plt.figure(figsize=(16*scale,8));
	ax = plt.subplot(1,2,1);
	ax.imshow(connect_mat_1,cmap="jet");
	
	if text:
		for ii in range(len(unique_pix)):
		    pos = np.where(connect_mat_1[:,:] == unique_pix[ii]);
		    pos0 = pos[0];
		    pos1 = pos[1];
		    ax.text((pos1)[np.array(len(pos1)/3,dtype=int)], (pos0)[np.array(len(pos0)/3,dtype=int)], f"{ii}",
		        verticalalignment='bottom', horizontalalignment='right',color='white', fontsize=15, fontweight="bold")
	ax.set(title="Superpixels")
	ax.title.set_fontsize(15)
	ax.title.set_fontweight("bold")
	
	ax1 = plt.subplot(1,2,2);
	dims = connect_mat_1.shape;
	connect_mat_1 = connect_mat_1.reshape(np.prod(dims),order="F");
	connect_mat_1[~np.in1d(connect_mat_1,pure_pix)]=0;
	connect_mat_1 = connect_mat_1.reshape(dims,order="F");

	ax1.imshow(connect_mat_1,cmap="jet");
	
	if text:
		for ii in range(len(pure_pix)):
		    pos = np.where(connect_mat_1[:,:] == pure_pix[ii]);
		    pos0 = pos[0];
		    pos1 = pos[1];
		    ax1.text((pos1)[np.array(len(pos1)/3,dtype=int)], (pos0)[np.array(len(pos0)/3,dtype=int)], f"{np.where(unique_pix==pure_pix[ii])[0][0]}",
		        verticalalignment='bottom', horizontalalignment='right',color='white', fontsize=15, fontweight="bold")
	ax1.set(title="Pure superpixels")
	ax1.title.set_fontsize(15)
	ax1.title.set_fontweight("bold")
	plt.tight_layout()
	plt.show();
	return fig

def superpixel_corr_plot(connect_mat_1, pure_pix, Cnt, brightness_rank, text=False):
	scale = np.maximum(1, (connect_mat_1.shape[1]/connect_mat_1.shape[0]));
	fig = plt.figure(figsize=(8*scale,16))
	ax = plt.subplot(2,1,1);
	dims = connect_mat_1.shape;
	connect_mat_1 = connect_mat_1.reshape(np.prod(dims),order="F");
	connect_mat_1[~np.in1d(connect_mat_1,pure_pix)]=0;
	connect_mat_1 = connect_mat_1.reshape(dims,order="F");

	ax.imshow(connect_mat_1,cmap="jet");	
	
	if text:
		for ii in range(len(pure_pix)):
		    pos = np.where(connect_mat_1[:,:] == pure_pix[ii]);
		    if len(pos[0]) > 0:
		    	pos0 = pos[0];
		    	pos1 = pos[1];
		    	ax.text((pos1)[np.array(len(pos1)/3,dtype=int)], (pos0)[np.array(len(pos0)/3,dtype=int)], f"{brightness_rank[ii]+1}",
		    	    verticalalignment='bottom', horizontalalignment='right',color='white', fontsize=15, fontweight="bold")
	ax.set(title="Pure superpixels")
	ax.title.set_fontsize(15)
	ax.title.set_fontweight("bold")
	
	ax1 = plt.subplot(2,1,2);
	show_img(ax1, Cnt);
	ax1.set(title="Local mean correlation")
	ax1.title.set_fontsize(15)
	ax1.title.set_fontweight("bold")
	plt.tight_layout()
	plt.show()
	return fig

def temporal_comp_plot(c, num_list=None, ini = False):
	num = c.shape[1];
	fig = plt.figure(figsize=(20,1.5*num))
	if num_list is None:
		num_list = np.arange(num);	
	for ii in range(num):
		plt.subplot(num,1, ii+1);
		plt.plot(c[:,ii]);
		if ii == 0:
			if ini:
				plt.title("Temporal components initialization for pure superpixels",fontweight="bold",fontsize=15);
			else:
				plt.title("Temporal components",fontweight="bold",fontsize=15);
		plt.ylabel(f"{num_list[ii]+1}",fontweight="bold",fontsize=15)
		if (ii > 0 and ii < num-1):
			plt.tick_params(axis='x',which='both',labelbottom='off')
		else:
			plt.xlabel("frames");
	plt.tight_layout()
	plt.show()
	return fig

def spatial_comp_plot(a, corr_img_all_r, num_list=None, ini=False):
	num = a.shape[1];
	patch_size = corr_img_all_r.shape[:2];
	scale = np.maximum(1, (corr_img_all_r.shape[1]/corr_img_all_r.shape[0]));
	fig = plt.figure(figsize=(8*scale,4*num));
	if num_list is None:
		num_list = np.arange(num);
	for ii in range(num):
		plt.subplot(num,2,2*ii+1);
		plt.imshow(a[:,ii].reshape(patch_size,order="F"),cmap='jet');
		plt.ylabel(str(num_list[ii]+1),fontsize=15,fontweight="bold");
		if ii==0:
			if ini:
				plt.title("Spatial components ini",fontweight="bold",fontsize=15);
			else:
				plt.title("Spatial components",fontweight="bold",fontsize=15);
		ax1 = plt.subplot(num,2,2*(ii+1));
		show_img(ax1, corr_img_all_r[:,:,ii]);
		if ii==0:
			ax1.set(title="corr image")
			ax1.title.set_fontsize(15)
			ax1.title.set_fontweight("bold")   
	plt.tight_layout()
	plt.show()
	return fig

def spatial_comp_plot2(a, corr_img_all_r, num_list=None, ini=False):
	num = a.shape[1];
	patch_size = corr_img_all_r.shape[:2];
	scale = np.maximum(1, (corr_img_all_r.shape[1]/corr_img_all_r.shape[0]));
	fig = plt.figure(figsize=(8*scale,4*num));
	if num_list is None:
		num_list = np.arange(num);
	for ii in range(num):
		plt.subplot(num,2,2*ii+1);
		plt.imshow(a[:,ii].reshape(patch_size,order="F"),cmap='jet');
		plt.ylabel(str(num_list[ii]+1),fontsize=15,fontweight="bold");
		if ii==0:
			if ini:
				plt.title("Spatial components ini",fontweight="bold",fontsize=15);
			else:
				plt.title("Spatial components",fontweight="bold",fontsize=15);
		plt.subplot(num,2,2*(ii+1));
		plt.imshow(corr_img_all_r[:,:,ii],cmap='jet');
		plt.title("Ground truth",fontweight="bold",fontsize=15);  
	plt.tight_layout()
	plt.show()
	return fig

def spatial_sum_plot(a, a_fin, patch_size, num_list_fin=None, text=False):
	scale = np.maximum(1, (patch_size[1]/patch_size[0]));
	fig = plt.figure(figsize=(16*scale,8));
	ax = plt.subplot(1,2,1);
	ax.imshow(a_fin.sum(axis=1).reshape(patch_size,order="F"),cmap="jet");
	
	if num_list_fin is None:
		num_list_fin = np.arange(a_fin.shape[1]);
	if text:
		for ii in range(a_fin.shape[1]):
			temp = a_fin[:,ii].reshape(patch_size,order="F");
			pos0 = np.where(temp == temp.max())[0][0];
			pos1 = np.where(temp == temp.max())[1][0];
			ax.text(pos1, pos0, f"{num_list_fin[ii]+1}", verticalalignment='bottom', horizontalalignment='right',color='white', fontsize=15, fontweight="bold")
	
	ax.set(title="more passes spatial components")
	ax.title.set_fontsize(15)
	ax.title.set_fontweight("bold")

	ax1 = plt.subplot(1,2,2);
	ax1.imshow(a.sum(axis=1).reshape(patch_size,order="F"),cmap="jet");
	
	if text:
		for ii in range(a.shape[1]):
			temp = a[:,ii].reshape(patch_size,order="F");
			pos0 = np.where(temp == temp.max())[0][0];
			pos1 = np.where(temp == temp.max())[1][0];
			ax1.text(pos1, pos0, f"{ii+1}", verticalalignment='bottom', horizontalalignment='right',color='white', fontsize=15, fontweight="bold")
	
	ax1.set(title="1 pass spatial components")
	ax1.title.set_fontsize(15)
	ax1.title.set_fontweight("bold")
	plt.tight_layout();
	plt.show()
	return fig

def spatial_sum_plot2(a_true, a_sup, a, a_fin, patch_size, num_list_fin=None, text=False):
	scale = np.maximum(1, (patch_size[1]/patch_size[0]));
	fig = plt.figure(figsize=(16*scale,16));
	ax2 = plt.subplot(2,2,1);
	ax2.imshow(a_sup.sum(axis=1).reshape(patch_size,order="F"),cmap="jet");
	
	if num_list_fin is None:
		num_list_fin = np.arange(a_sup.shape[1]);
	if text:
		for ii in range(a_sup.shape[1]):
			temp = a_sup[:,ii].reshape(patch_size,order="F");
			pos0 = np.where(temp == temp.max())[0][0];
			pos1 = np.where(temp == temp.max())[1][0];
			ax2.text(pos1, pos0, f"{num_list_fin[ii]+1}", verticalalignment='bottom', horizontalalignment='right',color='white', fontsize=15, fontweight="bold")
	
	ax2.set(title="Our method")#"more passes spatial components")
	ax2.title.set_fontsize(15)
	ax2.title.set_fontweight("bold")

	ax3 = plt.subplot(2,2,2);
	ax3.imshow(a_true.sum(axis=1).reshape(patch_size,order="F"),cmap="jet");
	
	if text:
		for ii in range(a_true.shape[1]):
			temp = a_true[:,ii].reshape(patch_size,order="F");
			pos0 = np.where(temp == temp.max())[0][0];
			pos1 = np.where(temp == temp.max())[1][0];
			ax3.text(pos1, pos0, f"{ii+1}", verticalalignment='bottom', horizontalalignment='right',color='white', fontsize=15, fontweight="bold")
	
	ax3.set(title="Ground truth")#"1 pass spatial components")
	ax3.title.set_fontsize(15)
	ax3.title.set_fontweight("bold")
	plt.tight_layout();
	plt.show()

	ax = plt.subplot(2,2,3);
	ax.imshow(a_fin.sum(axis=1).reshape(patch_size,order="F"),cmap="jet");
	
	if num_list_fin is None:
		num_list_fin = np.arange(a_fin.shape[1]);
	if text:
		for ii in range(a_fin.shape[1]):
			temp = a_fin[:,ii].reshape(patch_size,order="F");
			pos0 = np.where(temp == temp.max())[0][0];
			pos1 = np.where(temp == temp.max())[1][0];
			ax.text(pos1, pos0, f"{num_list_fin[ii]+1}", verticalalignment='bottom', horizontalalignment='right',color='white', fontsize=15, fontweight="bold")
	
	ax.set(title="NMF with ground truth initialization")#"more passes spatial components")
	ax.title.set_fontsize(15)
	ax.title.set_fontweight("bold")

	ax1 = plt.subplot(2,2,4);
	ax1.imshow(a.sum(axis=1).reshape(patch_size,order="F"),cmap="jet");
	
	if text:
		for ii in range(a.shape[1]):
			temp = a[:,ii].reshape(patch_size,order="F");
			pos0 = np.where(temp == temp.max())[0][0];
			pos1 = np.where(temp == temp.max())[1][0];
			ax1.text(pos1, pos0, f"{ii+1}", verticalalignment='bottom', horizontalalignment='right',color='white', fontsize=15, fontweight="bold")
	
	ax1.set(title="NMF with random initialization")#"1 pass spatial components")
	ax1.title.set_fontsize(15)
	ax1.title.set_fontweight("bold")
	plt.tight_layout();
	plt.show()
	return fig

def spatial_match_projection_plot(order, number, rlt_xya, rlt_yza, dims1, dims2):
	number = (order>=0).sum();
	scale = (dims1[1]+dims2[1])/max(dims1[0],dims2[0]);
	fig = plt.figure(figsize=(scale*2, 2*number));
	temp0 = np.where(order>=0)[0];
	temp1 = order[temp0];
	for ii in range(number):
		plt.subplot(number,2,2*ii+1);
		plt.imshow(rlt_xya[:,temp0[ii]].reshape(dims1[:2],order="F"),cmap="jet",aspect="auto");
		if ii == 0:
			plt.title("xy",fontsize=15,fontweight="bold");
			plt.ylabel("x",fontsize=15,fontweight="bold");
			plt.xlabel("y",fontsize=15,fontweight="bold");
		
		plt.subplot(number,2,2*ii+2);
		plt.imshow(rlt_yza[:,temp1[ii]].reshape(dims2[:2],order="F"),cmap="jet",aspect="auto");
		if ii == 0:
			plt.title("zy",fontsize=15,fontweight="bold");
			plt.ylabel("z",fontsize=15,fontweight="bold");
			plt.xlabel("y",fontsize=15,fontweight="bold");   
	plt.tight_layout()
	return fig

def spatial_compare_single_plot(a, patch_size):
	scale = (patch_size[1]/patch_size[0]);
	fig = plt.figure(figsize=(4*scale,4));
	plt.axis('off')
	plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
	plt.imshow(a.reshape(patch_size,order="F"),cmap='jet');
	plt.tight_layout();
	plt.show();
	return fig

def spatial_compare_nmf_plot(a, a_lasso_den, a_lasso_raw, order_Yd, order_Yraw, patch_size):
	num = a.shape[1];
	scale = (patch_size[1]/patch_size[0]);
	fig = plt.figure(figsize=(12*scale,4*num));

	for ii in range(num):
		ax0=plt.subplot(num,3,3*ii+1);
		plt.axis('off')
		plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
		img0=plt.imshow(a[:,ii].reshape(patch_size,order="F"),cmap='jet');
		if ii==0:
			plt.title("Our method",fontweight="bold",fontsize=15);

		ax1=plt.subplot(num,3,3*ii+2);
		if ii==0:
			plt.title("Sparse nmf on denoised data",fontweight="bold",fontsize=15);		
		if order_Yd[ii]>=0:
			img1=plt.imshow(a_lasso_den[:,order_Yd[ii]].reshape(patch_size,order="F"),cmap='jet');
		plt.axis('off')
		plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')

		ax2=plt.subplot(num,3,3*ii+3);
		if ii==0:
			plt.title("Sparse nmf on raw data",fontweight="bold",fontsize=15);					
		if order_Yraw[ii]>=0:
			img2=plt.imshow(a_lasso_raw[:,order_Yraw[ii]].reshape(patch_size,order="F"),cmap='jet');
		plt.axis('off')
		plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')

	plt.tight_layout()
	plt.show()
	return fig

def spatial_compare_nmf_gt_plot(a_gt, a, a_lasso_den, a_lasso_raw, order_Ys, order_Yd, order_Yraw, patch_size):
	num = a_gt.shape[1];
	scale = np.maximum(1, (patch_size[1]/patch_size[0]));
	fig = plt.figure(figsize=(16*scale,4*num));

	for ii in range(num):
		ax00=plt.subplot(num,4,4*ii+1);
		img00=plt.imshow(a_gt[:,ii].reshape(patch_size,order="F"),cmap='jet');
		plt.axis('off')
		plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
		if ii==0:
			plt.title("Ground truth",fontweight="bold",fontsize=15);

		ax0=plt.subplot(num,4,4*ii+2);
		if order_Ys[ii]>=0:
			img0=plt.imshow(a[:,order_Ys[ii]].reshape(patch_size,order="F"),cmap='jet');
		plt.axis('off')
		plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
		if ii==0:
			plt.title("Our method",fontweight="bold",fontsize=15);

		ax1=plt.subplot(num,4,4*ii+3);
		if ii==0:
			plt.title("Sparse nmf on denoised data",fontweight="bold",fontsize=15);		
		if order_Yd[ii]>=0:
			img1=plt.imshow(a_lasso_den[:,order_Yd[ii]].reshape(patch_size,order="F"),cmap='jet');
		plt.axis('off')
		plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')

		ax2=plt.subplot(num,4,4*ii+4);
		if ii==0:
			plt.title("Sparse nmf on raw data",fontweight="bold",fontsize=15);					
		if order_Yraw[ii]>=0:
			img2=plt.imshow(a_lasso_raw[:,order_Yraw[ii]].reshape(patch_size,order="F"),cmap='jet');
		plt.axis('off')
		plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')

	plt.tight_layout()
	plt.show()
	return fig

def temporal_compare_nmf_plot(c, c_lasso_den, c_lasso_raw, order_Yd, order_Yraw):
	num = c.shape[1];
	fig = plt.figure(figsize=(20,1.5*num))
	for ii in range(num):
		plt.subplot(num,1, ii+1);
		plt.plot(c[:,ii],label="our method");
		if order_Yd[ii]>=0:
			plt.plot(c_lasso_den[:,order_Yd[ii]],label="sparse nmf on denoised data");
		if order_Yraw[ii]>=0:
			plt.plot(c_lasso_raw[:,order_Yraw[ii]],label="sparse nmf on raw data");
		plt.legend();
		if ii == 0:
			plt.title("Temporal components",fontweight="bold",fontsize=15);
		plt.ylabel(f"{ii+1}",fontweight="bold",fontsize=15)
		if (ii > 0 and ii < num-1):
			plt.tick_params(axis='x',which='both',labelbottom='off')
		else:
			plt.xlabel("frames");
	plt.tight_layout()
	plt.show()
	return fig


def temporal_compare_plot(c, c_tf, ini = False):
	num = c.shape[1];
	fig = plt.figure(figsize=(20,1.5*num))
	for ii in range(num):
		plt.subplot(num,1, ii+1);
		plt.plot(c[:,ii],label="c");
		plt.plot(c_tf[:,ii],label="c_tf");
		plt.legend();
		if ii == 0:
			if ini:
				plt.title("Temporal components initialization for pure superpixels",fontweight="bold",fontsize=15);
			else:
				plt.title("Temporal components",fontweight="bold",fontsize=15);
		plt.ylabel(f"{ii+1}",fontweight="bold",fontsize=15)
		if (ii > 0 and ii < num-1):
			plt.tick_params(axis='x',which='both',labelbottom='off')
		else:
			plt.xlabel("frames");
	plt.tight_layout()
	plt.show()
	return fig

def temporal_compare_plot2(c, c_tf, ini = False):
	num = c.shape[1];
	fig = plt.figure(figsize=(20,1.5*num))
	for ii in range(num):
		plt.subplot(num,1, ii+1);
		plt.plot(c[:,ii],label="c");
		plt.plot(c_tf[:,ii],label="c true");
		plt.legend();
		if ii == 0:
			if ini:
				plt.title("Temporal components initialization for pure superpixels",fontweight="bold",fontsize=15);
			else:
				plt.title("Temporal components",fontweight="bold",fontsize=15);
		plt.ylabel(f"{ii+1}",fontweight="bold",fontsize=15)
		if (ii > 0 and ii < num-1):
			plt.tick_params(axis='x',which='both',labelbottom='off')
		else:
			plt.xlabel("frames");
	plt.tight_layout()
	plt.show()
	return fig

def show_img(ax, img,vmin=None,vmax=None):
	# Visualize local correlation, adapt from kelly's code
	im = ax.imshow(img,cmap='jet')
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.1)
	if np.abs(img.min())< 1:
		format_tile ='%.2f'
	else:
		format_tile ='%5d'
	plt.colorbar(im, cax=cax,orientation='vertical',spacing='uniform')

##################### vanilla nmf with random initialization with single penalty #########################
######### min|Y-UV|_2^2 + lambda*(|U|_1 + |V|_1) #####################
def vanilla_nmf_lasso(Yd, num_component, maxiter, tol, penalty_param, c=None):
	y0 = Yd.reshape(np.prod(Yd.shape[:2]),-1,order="F");
	if c is None:
		c = np.random.rand(y0.shape[1],num_component);
		c = c*np.sqrt(y0.mean()/num_component);

	clf_c = linear_model.Lasso(alpha=(penalty_param/(2*y0.shape[0])),positive=True,fit_intercept=False);
	clf_a = linear_model.Lasso(alpha=(penalty_param/(2*y0.shape[1])),positive=True,fit_intercept=True);
	res = np.zeros(maxiter);
	for iters in range(maxiter):
		temp = clf_a.fit(c, y0.T);
		a = temp.coef_;
		b = temp.intercept_;
		b = b.reshape(b.shape[0],1,order="F");
		c = clf_c.fit(a, y0-b).coef_;
		b = np.maximum(0, y0.mean(axis=1,keepdims=True)-(a*(c.mean(axis=0,keepdims=True))).sum(axis=1,keepdims=True));

		res[iters] = np.linalg.norm(y0 - np.matmul(a, c.T) - b,"fro")**2 + penalty_param*(abs(a).sum() + abs(c).sum());
		if iters > 0 and abs(res[iters] - res[iters-1])/res[iters-1] <= tol:
			break;
	if iters > 0:
		print(abs(res[iters] - res[iters-1])/res[iters-1]);

	temp = np.sqrt((a**2).sum(axis=0,keepdims=True));
	c = c*temp;
	a = a/temp;
	brightness = np.zeros(a.shape[1]);
	a_max = a.max(axis=0);
	c_max = c.max(axis=0);
	brightness = a_max * c_max;
	brightness_rank = np.argsort(-brightness); #a.shape[1] - ss.rankdata(brightness,method="ordinal");
	a = a[:,brightness_rank];
	c = c[:,brightness_rank];

	corr_img_all_r = a.copy();
	for ii in range(a.shape[1]):
		corr_img_all_r[:,ii] = vcorrcoef2(y0, c[:,ii]);
	#corr_img_all_r = np.corrcoef(y0,c.T)[:y0.shape[0],y0.shape[0]:];
	corr_img_all_r = corr_img_all_r.reshape(Yd.shape[0],Yd.shape[1],-1,order="F");
	return {"a":a, "c":c, "b":b, "res":res, "corr_img_all_r":corr_img_all_r}

def nnls_L0(X, Yp, noise):
	"""
	Nonnegative least square with L0 penalty, adapt from caiman
	It will basically call the scipy function with some tests
	we want to minimize :
	min|| Yp-W_lam*X||**2 <= noise
	with ||W_lam||_0  penalty
	and W_lam >0
	Parameters:
	---------
	    X: np.array
	        the input parameter ((the regressor
	    Y: np.array
	        ((the regressand
	Returns:
	--------
	    W_lam: np.array
	        the learned weight matrices ((Models
	"""
	W_lam, RSS = scipy.optimize.nnls(X, np.ravel(Yp))
	RSS = RSS * RSS
	if RSS > noise:  # hard noise constraint problem infeasible
		return W_lam

	print("hard noise constraint problem feasible!");
	while 1:
		eliminate = []
		for i in np.where(W_lam[:-1] > 0)[0]:  # W_lam[:-1] to skip background
			mask = W_lam > 0
			mask[i] = 0
			Wtmp, tmp = scipy.optimize.nnls(X * mask, np.ravel(Yp))
			if tmp * tmp < noise:
			    eliminate.append([i, tmp])
		if eliminate == []:
			return W_lam
		else:
			W_lam[eliminate[np.argmin(np.array(eliminate)[:, 1])][0]] = 0

def vanilla_nmf_multi_lasso(y0, num_component, maxiter, tol, fudge_factor=1, c_penalize=True, penalty_param=1e-4):
	sn = (noise_estimator(y0)**2)*y0.shape[1];
	c = np.random.rand(y0.shape[1],num_component);
	c = c*np.sqrt(y0.mean()/num_component);
	a = np.zeros([y0.shape[0],num_component]);
	res = np.zeros(maxiter);
	clf = linear_model.Lasso(alpha=penalty_param,positive=True,fit_intercept=False);
	for iters in range(maxiter):
		for ii in range(y0.shape[0]):
			a[ii,:] = nnls_L0(c, y0[[ii],:].T, fudge_factor * sn[ii]);
		if c_penalize:
			norma = (a**2).sum(axis=0);
			for jj in range(num_component):
				idx_ = np.setdiff1d(np.arange(num_component),ii);
				R_ = y0 - a[:,idx_].dot(c[:,idx_].T);
				V_ = (a[:,jj].T.dot(R_)/norma[jj]).reshape(1,y0.shape[1]);
				sv = (noise_estimator(V_)[0]**2)*y0.shape[1];
				c[:,jj] = nnls_L0(np.identity(y0.shape[1]), V_, fudge_factor * sv);
		else:
			#c = clf.fit(a, y0).coef_;
			c = np.maximum(0, np.matmul(np.matmul(np.linalg.inv(np.matmul(a.T,a)), a.T), y0)).T;    
		res[iters] = np.linalg.norm(y0 - np.matmul(a, c.T),"fro");
		if iters > 0 and abs(res[iters] - res[iters-1])/res[iters-1] <= tol:
			break;
	if iters > 0:
		print(abs(res[iters] - res[iters-1])/res[iters-1]);
	return a, c, res


def sim_noise(dims, noise_source):
	np.random.seed(0);
	N = np.prod(dims);
	noise_source = noise_source.reshape(np.prod(noise_source.shape), order="F");
	random_indices = np.random.randint(0, noise_source.shape[0], size=N);
	noise_sim = noise_source[random_indices].reshape(dims,order="F");
	return noise_sim


