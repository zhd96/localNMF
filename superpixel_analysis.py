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
import scipy
import merging
import cvxpy as cvx
import cvxopt as cvxopt

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

def show_img(ax, img,vmin=None,vmax=None):
    # Visualize local correlation, adapt from kelly's code
    im = ax.imshow(img,cmap='jet')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    if np.abs(img.min())< 1:
        format_tile ='%.2f'
    else:
        format_tile ='%5d'
    plt.colorbar(im, cax=cax,orientation='vertical',
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
                model = NMF(n_components=1, init='custom');
                U_mat[list(comp),ii] = model.fit_transform(y_temp, W=y_temp.mean(axis=1,keepdims=True),
                                                           H = y_temp.mean(axis=0,keepdims=True))[:,0];
                                                           V_mat[:,ii] = model.components_;
                                                           ii = ii+1;
        bg_comp_pos = np.where(U_mat.sum(axis=1) == 0)[0];
        bg_u, bg_s, bg_v = np.linalg.svd(Yt_r[bg_comp_pos,:],full_matrices=False);
        bg_u = bg_u[:,[0]];
        bg_v = bg_s[0]*bg_v[[0],:].T;
        bg_v = bg_v - bg_v.mean(axis=0,keepdims=True);
        if (bg_u[np.where(abs(bg_u)==abs(bg_u).max())[0], 0] < 0):
            bg_u = -1*bg_u;
            bg_v = -1*bg_v;
        bg_u = np.maximum(0, bg_u);
        bg_s = bg_s[0];
    
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

bg_comp_pos = np.where(U_mat.sum(axis=1) == 0)[0];
bg_u, bg_s, bg_v = np.linalg.svd(Yt_r[bg_comp_pos,:],full_matrices=False);
bg_u = bg_u[:,[0]];
bg_v = bg_s[0]*bg_v[[0],:].T;
bg_v = bg_v - bg_v.mean(axis=0,keepdims=True);
if (bg_u[np.where(abs(bg_u)==abs(bg_u).max())[0], 0] < 0):
    bg_u = -1*bg_u;
        bg_v = -1*bg_v;
        bg_u = np.maximum(0, bg_u);
        bg_s = bg_s[0];
    
    return V_mat, U_mat, B_mat, bg_u, bg_s, bg_v

#def vcorrcoef(X,Y):
#    """
#    Calculate correlation between X array and Y vector.
#
#    Parameters:
#    ----------------
#    X: 2d np.darray, dimension d1 x T
#    Y: 2d np.darray, dimension d2 x T
#
#    Return:
#    ----------------
#    r: 2d np.darray, dimension d1 x d2
#    correlation matrix
#    """
#    return np.corrcoef(X, Y)[:X.shape[0], X.shape[0]:]

def vcorrcoef(U, V, c):
    temp = (c - c.mean(axis=0,keepdims=True));
    return np.matmul(U, np.matmul(V - V.mean(axis=1,keepdims=True), temp/np.std(temp, axis=0, keepdims=True)));

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
    coef = np.matmul(np.matmul(np.linalg.inv(np.matmul(M[:,pure_pixels].T, M[:,pure_pixels])), M[:,pure_pixels].T), M);
    pure_pixels = np.array(pure_pixels);
    coef_rank = coef.copy(); ##### from large to small
for ii in range(len(pure_pixels)):
    coef_rank[:,ii] = [x for _,x in sorted(zip(len(pure_pixels) - ss.rankdata(coef[:,ii]), pure_pixels))];
    return pure_pixels, coef, coef_rank

def prepare_iteration(Yt, connect_mat_1, permute_col, unique_pix, pure_pixels, V_mat, U_mat):
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
    
    dims = Yt.shape;
    T = dims[2];
    
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
    a_ini = np.zeros([np.prod(dims[:-1]),len(pure_pix)]);
    for ii in range(len(pure_pix)):
        pos = np.where(permute_col==pure_pix[np.where(brightness_rank == ii+1)[0]])[0][0];
        c_ini[:,ii] = V_mat[:,pos];
        a_ini[:,ii] = U_mat[:,pos];
    #(connect_mat_1 == pure_pix[np.where(brightness_rank == ii+1)[0]])[left:(right+1),left:(right+1)].reshape((down-up+1)*(right-left+1),order="F");
    
    U, S, V = np.linalg.svd(Yt.reshape(np.prod(dims[:-1]),-1, order="F"), full_matrices=False);
    rank = sum(S > 1e-1);
    U = U[:,:rank]*S[:rank];
    V = V[:rank, :].T;

normalize_factor = np.std(np.matmul(U, V.T), axis=1, keepdims=True)*T;

#y0 = Yt[:dims[0],:dims[1]].reshape(np.prod(dims[:-1]),T,order="F");
corr_img_all_r = vcorrcoef(U/normalize_factor, V.T, c_ini).reshape(dims[0],dims[1],-1,order="F");
    
    return a_ini, c_ini, U, V, normalize_factor, brightness_rank, pure_pix, corr_img_all_r


def ls_solve_c(X, U, V, mask):
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
    beta_LS = np.matmul(np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), U), V.T);
    #beta_LS = np.vstack((np.maximum(0, temp[:-1,:]), temp[[-1],:]));
    #beta_LS = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), ind*Y);
    if mask is not None:
        beta_LS = beta_LS*mask;
    return beta_LS

def ls_solve_a(X, U, V, mask):
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
    beta_LS = np.maximum(0, np.matmul(np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), U), V.T));
    #beta_LS = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), ind*Y);
    if mask is not None:
        beta_LS = beta_LS*mask;
    return beta_LS

def make_mask(corr_img_all_r, corr, a, num_plane=1):
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
    unit_length = int(a.shape[0]/num_plane);
    corr_ini = corr;
    for ii in range(a.shape[1]):
        corr = corr_ini;
        while True:
            labeled_array, num_features = scipy.ndimage.measurements.label(corr_img_all_r[:,:,ii] > corr,structure=s);
            labeled_array = labeled_array.reshape(a.shape[0], order="F");
            ratio_use = np.array([]);
            
            for kk in range(num_plane):
                temp = np.unique(labeled_array[((kk)*unit_length):((kk+1)*unit_length)][a[((kk)*unit_length):((kk+1)*unit_length),ii]>0]); ########### which components does soma lies in ##############
                #print(temp);
                if sum(temp==0): ############ 0 means large background
                    temp = temp[1:];
                if len(temp) > 1:
                    ratio_temp = np.zeros(len(temp));
                    for jj in range(len(temp)):
                        ratio_temp[jj] = sum((np.in1d(labeled_array[((kk)*unit_length):((kk+1)*unit_length)],temp[jj]) > 0)*(a[((kk)*unit_length):((kk+1)*unit_length),ii]>0));
                    ratio_use = np.hstack([ratio_use, temp[np.where(ratio_temp == ratio_temp.max())[0]]]);
                else:
                    ratio_use = np.hstack([ratio_use, temp]);
        
            comp_pos = np.in1d(labeled_array, ratio_use) > 0;
            print(ratio_use)
            if sum(comp_pos)/(a.shape[0]) < 0.3:
                break;
    else:
        print("corr too low!")
        corr = corr + 0.1;
        mask_a[:,ii] = comp_pos;
    return mask_a

def merge_components(a,c,merge_corr_thr=0.8,merge_overlap_thr=0.8):
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
    ########### calculate overlap area ###########
    a_corr = np.matmul(((a>0)*1).T, (a>0)*1);
    a_corr = a_corr/(((a>0)*1).sum(axis=0,keepdims=True));
    cri1 = (((a_corr > merge_overlap_thr)+(a_corr.T > merge_overlap_thr))*1 > 0); ########### spatially highly overlapped
    cri2 = (np.corrcoef(c.T) > merge_corr_thr)*(a_corr > 0); ########### temporally high correlated
    
    cri = (cri1+cri2);
    cri = np.triu(cri, k=1);
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
        print("merge" + str(comps));
        for comp in comps:
            comp=list(comp);
            a_zero = np.zeros([a.shape[0],1]);
            a_temp = a[:,comp];
            mask_temp = np.where(a_temp.sum(axis=1,keepdims=True) > 0)[0];
            a_temp = a_temp[mask_temp,:];
            y_temp = np.matmul(a_temp, c[:,comp].T);
            a_temp = a_temp.mean(axis=1,keepdims=True);
            c_temp = c[:,comp].mean(axis=1,keepdims=True);
            model = NMF(n_components=1, init='custom')
            a_temp = model.fit_transform(y_temp, W=a_temp, H = (c_temp.T));
            a_zero[mask_temp] = a_temp;
            c_temp = model.components_.T;
            
            a_pri = np.hstack((a_pri,a_zero));
            c_pri = np.hstack((c_pri,c_temp));
        return flag, a_pri, c_pri
    else:
        flag = 0;
        return flag

def update_AC_l2(U, V, normalize_factor, a, c, ff, patch_size, corr_th_fix,
                 maxiter=50, tol=1e-8, update_after=None,merge_corr_thr=0.5,merge_overlap_thr=0.7, num_plane=1):
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
    
    ###################### initialize A ##########################
    
    mask_a = (a > 0);
    #mask_a = (corr_img_all > corr_th_ini);
    mask_ab = np.hstack((mask_a,g));
    temp = ls_solve_a(np.hstack((c,f,ff)), V, U, mask_ab.T).T;
    a = temp[:,:-num_bg];
    b = temp[:,[-num_bg]];
    fb = temp[:,(-num_bg+1):];
    
    try:
        temp = ls_solve_c(np.hstack((a,fb)), np.hstack((U,b)), np.hstack((V,-1*f)), None).T;
        c = temp[:,:(-num_bg+1)];
        ff = temp[:,(-num_bg+1):];
        c = c - c.min(axis=0,keepdims=True);
        ff = ff - ff.mean(axis=0,keepdims=True);
        b = np.maximum(0, (U*(V.mean(axis=0,keepdims=True))).sum(axis=1,keepdims=True)-(a*(c.mean(axis=0,keepdims=True))).sum(axis=1,keepdims=True));
    
    except:
        print("zero a!");
        pos = np.where(a.sum(axis=0) == 0)[0];
        print("delete components" + str(pos));
        a = np.delete(a, pos, axis=1);
        #mask_c = np.delete(mask_c, pos, axis=1);
        temp = ls_solve_c(np.hstack((a,fb)), np.hstack((U,b)), np.hstack((V,-1*f)), None).T;
        c = temp[:,:(-num_bg+1)];
        ff = temp[:,(-num_bg+1):];
        c = c - c.min(axis=0,keepdims=True);
        ff = ff - ff.mean(axis=0,keepdims=True);
        b = np.maximum(0, (U*(V.mean(axis=0,keepdims=True))).sum(axis=1,keepdims=True)-(a*(c.mean(axis=0,keepdims=True))).sum(axis=1,keepdims=True));

##################### iteratively update A and C (dilate A) ##############################
########## update mask a #################
corr_img_all_r = vcorrcoef(U/normalize_factor, V.T, c).reshape(patch_size[0],patch_size[1],-1,order="F");
mask_a = make_mask(corr_img_all_r, corr_th_fix, a, num_plane);
if sum(mask_a.sum(axis=0) == 0):
    print("zero mask a!");
    pos = np.where(mask_a.sum(axis=0) == 0)[0];
    print("delete components" + str(pos));
    mask_a = np.delete(mask_a, pos, axis=1);
    a = np.delete(a, pos, axis=1);
    #mask_c = np.delete(mask_c, pos, axis=1);
        c = np.delete(c, pos, axis=1);
    mask_ab = np.hstack((mask_a,g));

for iters in range(maxiter):
    try:
        temp = ls_solve_a(np.hstack((c,f,ff)), V, U, mask_ab.T).T;
        a = temp[:,:-num_bg];
        b = temp[:,[-num_bg]];
        fb = temp[:,(-num_bg+1):];
        except:
            print("zero c!");
            pos = np.where(c.sum(axis=0) == 0)[0];
            print("delete components" + str(pos));
            mask_a = np.delete(mask_a, pos, axis=1);
            mask_ab = np.hstack((mask_a,g));
            c = np.delete(c, pos, axis=1);
            #mask_c = np.delete(mask_c, pos, axis=1);
            temp = ls_solve_a(np.hstack((c,f,ff)), V, U, mask_ab.T).T;
            a = temp[:,:-num_bg];
            b = temp[:,[-num_bg]];
            fb = temp[:,(-num_bg+1):];
        try:
            temp = ls_solve_c(np.hstack((a,fb)), np.hstack((U,b)), np.hstack((V,-1*f)), None).T;
            c = temp[:,:(-num_bg+1)];
            ff = temp[:,(-num_bg+1):];
            c = c - c.min(axis=0,keepdims=True);
            ff = ff - ff.mean(axis=0,keepdims=True);
            b = np.maximum(0, (U*(V.mean(axis=0,keepdims=True))).sum(axis=1,keepdims=True)-(a*(c.mean(axis=0,keepdims=True))).sum(axis=1,keepdims=True));
    
        except:
            print("zero a!");
            pos = np.where(a.sum(axis=0) == 0)[0];
            print("delete components" + str(pos));
            mask_a = np.delete(mask_a, pos, axis=1);
            mask_ab = np.hstack((mask_a,g));
            a = np.delete(a, pos, axis=1);
            #mask_c = np.delete(mask_c, pos, axis=1);
            temp = ls_solve_c(np.hstack((a,fb)), np.hstack((U,b)), np.hstack((V,-1*f)), None).T;
            c = temp[:,:(-num_bg+1)];
            ff = temp[:,(-num_bg+1):];
            c = c - c.min(axis=0,keepdims=True);
            ff = ff - ff.mean(axis=0,keepdims=True);
            b = np.maximum(0, (U*(V.mean(axis=0,keepdims=True))).sum(axis=1,keepdims=True)-(a*(c.mean(axis=0,keepdims=True))).sum(axis=1,keepdims=True));
        
        # Merge Components
        if update_after and ((iters + 1) % update_after == 0):
            rlt = merge_components(a,c,merge_corr_thr=merge_corr_thr,merge_overlap_thr=merge_overlap_thr);
            flag = isinstance(rlt, int);
            if ~np.array(flag):
                a = rlt[1];
                c = rlt[2];
                mask_c = np.ones(c.shape);
            else:
                print("no merge!")
            corr_img_all_r = vcorrcoef(U/normalize_factor, V.T, c).reshape(patch_size[0],patch_size[1],-1,order="F");
            mask_a = make_mask(corr_img_all_r, corr_th_fix, a, num_plane);
            #if ~np.array(flag):
            #spatial_comp_plot(a, corr_img_all_r, ini=False);
            if sum(mask_a.sum(axis=0) == 0):
                print("zero mask a!");
                pos = np.where(mask_a.sum(axis=0) == 0)[0];
                print("delete components" + str(pos));
                mask_a = np.delete(mask_a, pos, axis=1);
                a = np.delete(a, pos, axis=1);
                #mask_c = np.delete(mask_c, pos, axis=1);
                c = np.delete(c, pos, axis=1);

mask_ab = np.hstack((mask_a,g));
#print(np.matmul(fb, ff.T).shape);
residual = (np.matmul(U, V.T) - np.matmul(a, c.T) - b - np.matmul(fb, ff.T));
    res[iters] = np.linalg.norm(residual, "fro");
    if iters > 0:
        if abs(res[iters] - res[iters-1])/res[iters-1] <= tol:
            break;
    c = c*np.sqrt((a**2).sum(axis=0,keepdims=True));
    a = a/np.sqrt((a**2).sum(axis=0,keepdims=True));
    brightness = np.zeros(a.shape[1]);
    for ii in range(a.shape[1]):
        v_max = a[:,ii].max();
        u_max = c[:,ii].max();
        brightness[ii] = u_max * v_max;
    brightness_rank = a.shape[1] - ss.rankdata(brightness,method="ordinal");
    #brightness_rank = np.arange(a.shape[1]);
    a_cp = a.copy();
    c_cp = c.copy();
    corr_cp = corr_img_all_r.copy();
    for ii in range(a.shape[1]):
        a_cp[:,ii] = a[:,np.where(brightness_rank==ii)[0][0]];
        c_cp[:,ii] = c[:,np.where(brightness_rank==ii)[0][0]];
        corr_cp[:,:,ii] = corr_img_all_r[:,:,np.where(brightness_rank==ii)[0][0]];
    if iters > 0:
        print("residual relative change: " + str(abs(res[iters] - res[iters-1])/res[iters-1]));
return a_cp, c_cp, b, fb, ff, res, corr_cp


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
    non_zero_num = sum(sum(mask));
    if beta is None:
        beta = beta_LS*mask;
    for jj in range(maxiter_nr):
        beta0 = beta.copy();
        beta = np.maximum(0, beta_LS - np.matmul(X_plus, ind*np.maximum(0, Y - np.matmul(X, beta) - mu)));
        beta = beta*mask;
        res = np.linalg.norm(beta0 - beta, "fro")/non_zero_num;
        print(res);
        if jj > 0:
            if res <= tol_nr:
                print(jj);
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

def update_AC_huber(y0, c_ini, corr_img_all_r, corr_th_ini, corr_th_fix, corr_th_dilate, dilate_times=2,
                    mu=None, maxiter=60, tol=1e-8, maxiter_nr=100, tol_nr=1, whole_data=True,merge_after=None,
                    merge_corr_thr=0.45,merge_overlap_thr=0.7):
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

if mu is None: ######### if mu is none, use different change point for each pixel.
    noise_std_ = noise_estimator(y0);
    mu_vec = noise_std_.reshape(y0.shape[0],1);#np.median(noise_std_);
    mu = 0.5;
        y0 = y0/mu_vec;
    else:
        mu_vec = None;
    
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
        # Merge Components
        if merge_after and ((iters + 1) % merge_after == 0):
            rlt = merge_components(a,c,merge_corr_thr=merge_corr_thr,merge_overlap_thr=merge_overlap_thr);
            flag = isinstance(rlt, int);
            if ~np.array(flag):
                a = rlt[1];
                c = rlt[2];
                mask_c = np.ones(c.shape);
                corr_img_all = vcorrcoef(y0, c.T);
                corr_img_all_r = corr_img_all.reshape(corr_img_all_r.shape[0],corr_img_all_r.shape[1],-1,order="F");
                mask_a = make_mask(corr_img_all_r, corr_th_fix, a);
                mask_ab = np.hstack((mask_a,g));
            else:
                print("no merge!")

residual = ind*(y0 - np.matmul(a, c.T) - b);
res[iters] = np.sqrt(one_side_huber_loss(residual, mu));
if iters > 0:
    if abs(res[iters] - res[iters-1])/res[iters-1] <= tol:
        break;
    if iters > 0:
        print(abs(res[iters] - res[iters-1])/res[iters-1]);

if mu_vec is not None:
    a = a*mu_vec;
        b = b*mu_vec;
        a_ini = a_ini*mu_vec;
    
    return a, a_ini, c, c_ini, b, res, mu_vec, corr_img_all_r


##################### vanilla nmf with random initialization with single penalty #########################
######### min|Y-UV|_2^2 + lambda*(|U|_1 + |V|_1) #####################
def vanilla_nmf_lasso(y0, num_component, maxiter, tol, penalty_param):
    c = np.random.rand(y0.shape[1],num_component);
    c = c*np.sqrt(y0.mean()/num_component);
    clf = linear_model.Lasso(alpha=penalty_param,positive=True,fit_intercept=False);
    res = np.zeros(maxiter);
    for iters in range(maxiter):
        a = clf.fit(c, y0.T).coef_;
        c = clf.fit(a, y0).coef_;
        
        res[iters] = np.linalg.norm(y0 - np.matmul(a, c.T),"fro");
        if iters > 0 and abs(res[iters] - res[iters-1])/res[iters-1] <= tol:
            break;
    if iters > 0:
        print(abs(res[iters] - res[iters-1])/res[iters-1]);
    return a, c, res

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

def reconstruct(Yd, spatial_components, temporal_components, background_components):
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
    mov_res = y0 - (np.matmul(spatial_components, temporal_components.T)+background_components).reshape(dims, order='F');
    return mov_res

def pure_superpixel_compare_plot(connect_mat_1, unique_pix, pure_pixels):
    scale = np.maximum(1, int(connect_mat_1.shape[1]/connect_mat_1.shape[0]));
    fig = plt.figure(figsize=(16*scale,8));
    ax = plt.subplot(1,2,1);
    ax.imshow(connect_mat_1,cmap="jet");
    
    for ii in range(len(unique_pix)):
        pos = np.where(connect_mat_1[:,:] == unique_pix[ii]);
        pos0 = pos[0];
        pos1 = pos[1];
        ax.text((pos1)[np.array(len(pos1)/3,dtype=int)], (pos0)[np.array(len(pos0)/3,dtype=int)], f"{ii}",
                verticalalignment='bottom', horizontalalignment='right',color='white', fontsize=15, fontweight="bold")
    ax.set(title="Superpixels in this patch")
    ax.title.set_fontsize(15)
    ax.title.set_fontweight("bold")
    
    ax1 = plt.subplot(1,2,2);
    ax1.imshow(connect_mat_1,cmap="jet");
    
    for ii in range(len(pure_pixels)):
        pos = np.where(connect_mat_1[:,:] == unique_pix[pure_pixels[ii]]);
        pos0 = pos[0];
        pos1 = pos[1];
        ax1.text((pos1)[np.array(len(pos1)/3,dtype=int)], (pos0)[np.array(len(pos0)/3,dtype=int)], f"{pure_pixels[ii]}",
                 verticalalignment='bottom', horizontalalignment='right',color='white', fontsize=15, fontweight="bold")
ax1.set(title="Pure superpixels in this patch")
ax1.title.set_fontsize(15)
ax1.title.set_fontweight("bold")
plt.tight_layout()
plt.show();
return fig

def superpixel_corr_plot(connect_mat_1, pure_pix, Cnt, brightness_rank):
    scale = np.maximum(1, int(connect_mat_1.shape[1]/connect_mat_1.shape[0]));
    fig = plt.figure(figsize=(16*scale,8))
    ax = plt.subplot(1,2,1);
    ax.imshow(connect_mat_1,cmap="jet");
    
    for ii in range(len(pure_pix)):
        pos = np.where(connect_mat_1[:,:] == pure_pix[ii]);
        if len(pos[0]) > 0:
            pos0 = pos[0];
            pos1 = pos[1];
            ax.text((pos1)[np.array(len(pos1)/3,dtype=int)], (pos0)[np.array(len(pos0)/3,dtype=int)], f"{brightness_rank[ii]}",
                    verticalalignment='bottom', horizontalalignment='right',color='white', fontsize=15, fontweight="bold")
    ax.set(title="Pure superpixels in this patch")
    ax.title.set_fontsize(15)
    ax.title.set_fontweight("bold")
    
    ax1 = plt.subplot(1,2,2);
    show_img(ax1, Cnt);
    ax1.set(title="Local mean correlation for Yt")
    ax1.title.set_fontsize(15)
    ax1.title.set_fontweight("bold")
    plt.tight_layout()
    plt.show()
    return fig

def temporal_comp_plot(c, ini = False):
    num = c.shape[1];
    fig = plt.figure(figsize=(20,1.5*num))
    for ii in range(num):
        plt.subplot(num,1, ii+1);
        plt.plot(c[:,ii]);
        if ii == 0:
            if ini:
                plt.title("Temporal components initialization for pure superpixel in this patch",fontweight="bold",fontsize=15);
            else:
                plt.title("Temporal components in this patch",fontweight="bold",fontsize=15);
        plt.ylabel(f"{ii+1}",fontweight="bold",fontsize=15)
        if (ii > 0 and ii < num-1):
            plt.tick_params(axis='x',which='both',labelbottom='off')
        else:
            plt.xlabel("frames");
    plt.tight_layout()
    plt.show()
    return fig

def spatial_comp_plot(a, corr_img_all_r, ini=False):
    num = a.shape[1];
    patch_size = corr_img_all_r.shape[:2];
    scale = np.maximum(1, int(corr_img_all_r.shape[1]/corr_img_all_r.shape[0]));
    fig = plt.figure(figsize=(8*scale,4*num));
    for ii in range(num):
        plt.subplot(num,2,2*ii+1);
        plt.imshow(a[:,ii].reshape(patch_size,order="F"),cmap='jet');
        plt.ylabel(str(ii+1),fontsize=15,fontweight="bold");
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

def spatial_sum_plot(a, a_fin, patch_size, text=True):
    scale = np.maximum(1, int(patch_size[1]/patch_size[0]));
    fig = plt.figure(figsize=(16*scale,8));
    ax = plt.subplot(1,2,1);
    ax.imshow(a_fin.sum(axis=1).reshape(patch_size,order="F"),cmap="jet");
    
    if text:
        for ii in range(a_fin.shape[1]):
            temp = a_fin[:,ii].reshape(patch_size,order="F");
            pos0 = np.where(temp == temp.max())[0][0];
            pos1 = np.where(temp == temp.max())[1][0];
            ax.text(pos1, pos0, f"{ii+1}", verticalalignment='bottom', horizontalalignment='right',color='white', fontsize=15, fontweight="bold")
    
    ax.set(title="Two passes")
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
    
    ax1.set(title="One pass")
    ax1.title.set_fontsize(15)
    ax1.title.set_fontweight("bold")
    plt.tight_layout();
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
                plt.title("Temporal components initialization for pure superpixel in this patch",fontweight="bold",fontsize=15);
            else:
                plt.title("Temporal components in this patch",fontweight="bold",fontsize=15);
        plt.ylabel(f"{ii+1}",fontweight="bold",fontsize=15)
        if (ii > 0 and ii < num-1):
            plt.tick_params(axis='x',which='both',labelbottom='off')
        else:
            plt.xlabel("frames");
    plt.tight_layout()
    plt.show()
    return fig

## 0.05 for Y_orig
def axon_pipeline(Yd, cut_off_point=0.95, length_cut=10, residual_cut = 0.6, corr_th_fix=0.5,
                  merge_corr_thr=0.8, merge_overlap_thr=0.8, refine = True, num_plane=1,
                  cut_off_point2=0.95, cut_off_point3=[0.9,0.9], plot_en=False, more_pass=0,
                  TF=False, fudge_factor=1): #length_cut2=10, residual_cut2 = 0.6,
    #merge_corr_thr_fin = 0.8, merge_overlap_thr_fin = 0.8, plot_en=False):
    
    patch_size = Yd.shape[:2];
    #Yd = Yd - Yd.min(axis=2, keepdims=True);
    ############ first pass ##################
    print("start first pass!")
    #Yd = Yd[up:down, left:right,:];
    Yt = threshold_data(Yd, th=2);
    if num_plane > 1:
        print("3d data!");
        connect_mat_1, idx, comps, permute_col = find_superpixel_3d(Yt,num_plane,cut_off_point,length_cut,eight_neighbours=True);
    else:
        connect_mat_1, idx, comps, permute_col = find_superpixel(Yt,cut_off_point,length_cut,eight_neighbours=True);
    Cnt = local_correlations_fft(Yt);
    V_mat, U_mat, B_mat, bg_u, bg_s, bg_v = spatial_temporal_ini(Yt, comps, idx, length_cut, maxiter=5, whole_data=True, method="svd");
    unique_pix, M = search_superpixel_in_range(connect_mat_1, permute_col, V_mat);
    pure_pixels, coef, coef_rank = fast_sep_nmf(M, M.shape[1], residual_cut);

if plot_en:
    pure_superpixel_compare_plot(connect_mat_1, unique_pix, pure_pixels);
    a_ini, c_ini, U, V, normalize_factor, brightness_rank, pure_pix, corr_img_all_r = prepare_iteration(Yd, connect_mat_1, permute_col, unique_pix, pure_pixels, V_mat, U_mat);
    #return bg_u, bg_v, bg_s
    
    if plot_en:
        superpixel_corr_plot(connect_mat_1, pure_pix, Cnt, brightness_rank);
#return connect_mat_1, pure_pix, Cnt, brightness_rank
#temporal_comp_plot(c_ini,ini = True);
#spatial_comp_plot(a_ini, corr_img_all_r, ini=True);
#return coef_rank
print("start first pass iteration!")
a, c, b, fb, ff, res, corr_img_all_r = update_AC_l2(U, V, normalize_factor, a_ini, c_ini, bg_v, patch_size,
                                                    corr_th_fix, maxiter=50, tol=1e-8, update_after=5,
                                                    merge_corr_thr=merge_corr_thr,merge_overlap_thr=merge_overlap_thr, num_plane=num_plane);
#print(res)
if plot_en:
    temporal_comp_plot(c,ini = False);
    spatial_comp_plot(a, corr_img_all_r, ini=False);
    ############# second pass #################
    if refine:
        print("start second pass!");
        Yd_res = reconstruct(Yd, a, c, b);
        Yt_res = threshold_data(Yd_res,th=1);
        if num_plane > 1:
            print("3d data!");
            connect_mat_1_res, idx_res, comps_res, permute_col_res = find_superpixel_3d(Yt_res,num_plane,cut_off_point2,length_cut,eight_neighbours=True);
        else:
            connect_mat_1_res, idx_res, comps_res, permute_col_res = find_superpixel(Yt_res,cut_off_point2,length_cut,eight_neighbours=True);
        Cnt_res = local_correlations_fft(Yt_res);
        V_mat_res, U_mat_res, B_mat_res, bg_u_res, bg_s_res, bg_v_res = spatial_temporal_ini(Yt_res, comps_res, idx_res, length_cut, maxiter=5, whole_data=True, method="svd");
        unique_pix_res, M_res = search_superpixel_in_range(connect_mat_1_res, permute_col_res, V_mat_res);
        pure_pixels_res, coef_res, coef_rank_res = fast_sep_nmf(M_res, M_res.shape[1], residual_cut);
        if plot_en:
            pure_superpixel_compare_plot(connect_mat_1_res, unique_pix_res, pure_pixels_res);
        a_ini_res, c_ini_res, U_res, V_res, normalize_factor_res, brightness_rank_res, pure_pix_res, corr_img_all_r_res = prepare_iteration(Yd_res, connect_mat_1_res, permute_col_res, unique_pix_res, pure_pixels_res, V_mat_res, U_mat_res);
        if plot_en:
            superpixel_corr_plot(connect_mat_1_res, pure_pix_res, Cnt_res, brightness_rank_res);
        #temporal_comp_plot(c_ini_res,ini = True);
        #spatial_comp_plot(a_ini_res, corr_img_all_r_res, ini=True);
        print("start second pass iteration!")
        #return {'a_fin':a_ini_res, 'c_fin':c_ini_res, 'y0':y0_res}
        a_res, c_res, b_res, fb_res, ff_res, res_res, corr_img_all_r_res = update_AC_l2(U_res, V_res, normalize_factor_res, a_ini_res, c_ini_res, bg_v_res, patch_size,
                                                                                        corr_th_fix, maxiter=50, tol=1e-8, update_after=5,
                                                                                        merge_corr_thr=merge_corr_thr,merge_overlap_thr=merge_overlap_thr, num_plane=num_plane);
                                                                                        if plot_en:
                                                                                            #temporal_comp_plot(c_res,ini = False);
                                                                                            spatial_comp_plot(a_res, corr_img_all_r_res, ini=False);
                                                                                                a_ini_fin = np.hstack((a, a_res));
                                                                                                c_ini_fin = np.hstack((c, c_res));
                                                                                                bg_v_fin = np.hstack((bg_v, bg_v_res));
                                                                                                
                                                                                                ii = 0;
                                                                                                    while ii < more_pass:
                                                                                                        print("start " + str(ii+3) + " pass!");
                                                                                                        Yd_res = reconstruct(Yd_res, a_res, c_res, b_res);
                                                                                                        Yt_res = threshold_data(Yd_res,th=0.5);
                                                                                                        if num_plane > 1:
                                                                                                            print("3d data!");
                                                                                                            connect_mat_1_res, idx_res, comps_res, permute_col_res = find_superpixel_3d(Yt_res,num_plane,cut_off_point3[ii],length_cut,eight_neighbours=True);
                                                                                                                else:
                                                                                                                    connect_mat_1_res, idx_res, comps_res, permute_col_res = find_superpixel(Yt_res,cut_off_point3[ii],length_cut,eight_neighbours=True);
                                                                                                                        Cnt_res = local_correlations_fft(Yt_res);
                                                                                                                        V_mat_res, U_mat_res, B_mat_res, bg_u_res, bg_s_res, bg_v_res = spatial_temporal_ini(Yt_res, comps_res, idx_res, length_cut, maxiter=5, whole_data=True, method="svd");
                                                                                                                        unique_pix_res, M_res = search_superpixel_in_range(connect_mat_1_res, permute_col_res, V_mat_res);
                                                                                                                        pure_pixels_res, coef_res, coef_rank_res = fast_sep_nmf(M_res, M_res.shape[1], residual_cut);
                                                                                                                        if plot_en:
                                                                                                                            pure_superpixel_compare_plot(connect_mat_1_res, unique_pix_res, pure_pixels_res);
                                                                                                                                a_ini_res, c_ini_res, U_res, V_res, normalize_factor_res, brightness_rank_res, pure_pix_res, corr_img_all_r_res = prepare_iteration(Yd_res, connect_mat_1_res, permute_col_res, unique_pix_res, pure_pixels_res, V_mat_res, U_mat_res);
                                                                                                                                if plot_en:
                                                                                                                                    superpixel_corr_plot(connect_mat_1_res, pure_pix_res, Cnt_res, brightness_rank_res);
                                                                                                                                        #temporal_comp_plot(c_ini_res,ini = True);
                                                                                                                                        #spatial_comp_plot(a_ini_res, corr_img_all_r_res, ini=True);
                                                                                                                                        #return y0_res, a_ini_res, c_ini_res, bg_v_res
                                                                                                                                        print("start " + str(ii+3) + " pass iteration!")
                                                                                                                                        #return {'a_fin':a_ini_res, 'c_fin':c_ini_res, 'y0':y0_res}
                                                                                                                                        a_res, c_res, b_res, fb_res, ff_res, res_res, corr_img_all_r_res = update_AC_l2(U_res, V_res, normalize_factor_res, a_ini_res, c_ini_res, bg_v_res, patch_size,
                                                                                                                                                                                                                        corr_th_fix, maxiter=50, tol=1e-8, update_after=5,
                                                                                                                                                                                                                        merge_corr_thr=merge_corr_thr,merge_overlap_thr=merge_overlap_thr, num_plane=num_plane);
                                                                                                                                                                                                                        if plot_en:
                                                                                                                                                                                                                            #temporal_comp_plot(c_res,ini = False);
                                                                                                                                                                                                                            spatial_comp_plot(a_res, corr_img_all_r_res, ini=False);
                                                                                                                                                                                                                                a_ini_fin = np.hstack((a_ini_fin, a_res));
                                                                                                                                                                                                                                c_ini_fin = np.hstack((c_ini_fin, c_res));
                                                                                                                                                                                                                                bg_v_fin = np.hstack((bg_v_fin, bg_v_res));
                                                                                                                                                                                                                                
                                                                                                                                                                                                                                    ii = ii+1;
                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                        print("final update!");
                                                                                                                                                                                                                                        #return a_ini_fin, c_ini_fin, U, V, normalize_factor, bg_v_fin
                                                                                                                                                                                                                                        a_fin, c_fin, b_fin, fb_fin, ff_fin, res_fin, corr_img_all_r_fin = update_AC_l2(U, V, normalize_factor, a_ini_fin, c_ini_fin, bg_v_fin, patch_size,
                                                                                                                                                                                                                                                                                                                        corr_th_fix, maxiter=30, tol=1e-8, update_after=2,
                                                                                                                                                                                                                                                                                                                        merge_corr_thr=merge_corr_thr,merge_overlap_thr=merge_overlap_thr, num_plane=num_plane)
                                                                                                                                                                                                                                                                                                                        c_fin_tf = [];
                                                                                                                                                                                                                                                                                                                            if TF:
                                                                                                                                                                                                                                                                                                                                c_fin_tf = c_fin.copy();
                                                                                                                                                                                                                                                                                                                                sigma = noise_estimator(c_fin.T);
                                                                                                                                                                                                                                                                                                                                sigma *= fudge_factor
                                                                                                                                                                                                                                                                                                                                    for ii in range(c_fin.shape[1]):
                                                                                                                                                                                                                                                                                                                                        c_fin_tf[:,ii] = l1_tf(c_fin[:,ii], sigma[ii]);
                                                                                                                                                                                                                                                                                                                                            if plot_en:
                                                                                                                                                                                                                                                                                                                                                temporal_compare_plot(c_fin,c_fin_tf, ini = False);
                                                                                                                                                                                                                                                                                                                                                    else:
                                                                                                                                                                                                                                                                                                                                                        if plot_en:
                                                                                                                                                                                                                                                                                                                                                            temporal_comp_plot(c_fin,ini = False);
                                                                                                                                                                                                                                                                                                                                                                
                                                                                                                                                                                                                                                                                                                                                                if plot_en:
                                                                                                                                                                                                                                                                                                                                                                    spatial_comp_plot(a_fin, corr_img_all_r_fin, ini=False);
                                                                                                                                                                                                                                                                                                                                                                    spatial_sum_plot(a, a_fin, patch_size);
                                                                                                                                                                                                                                                                                                                                                                        return {'a':a, 'c':c, 'b':b, 'corr_img_all_r':corr_img_all_r, 'a_fin':a_fin, 'c_fin':c_fin, 'c_fin_tf':c_fin_tf, 'b_fin':b_fin, 'fb':fb, 'ff':ff, 'fb_fin':fb_fin, 'ff_fin':ff_fin, 'corr_img_all_r_fin':corr_img_all_r_fin, 'connect_mat_1':connect_mat_1_res, 'pure_pix':pure_pix_res, 'Cnt':Cnt_res, 'brightness_rank':brightness_rank_res}
                                                                                                                                                                                                                                                                                                                                                                    else:
                                                                                                                                                                                                                                                                                                                                                                        return {'a':a, 'c':c, 'b':b, 'fb':fb, 'ff':ff, 'corr_img_all_r':corr_img_all_r, 'connect_mat_1':connect_mat_1, 'pure_pix':pure_pix, 'Cnt':Cnt, 'brightness_rank':brightness_rank}

def l1_tf(y, sigma):
    if np.abs(sigma/y.max())<=1e-3:
        print('Do not denoise (high SNR: noise_level=%.3e)'%sigma) if verbose else 0
        return y
    n = y.size
    # Form second difference matrix.
    D = (np.diag(2*np.ones(n),0)+np.diag(-1*np.ones(n-1),1)+np.diag(-1*np.ones(n-1),-1))[1:n-1];
    x = cvx.Variable(n)
    obj = cvx.Minimize(cvx.norm(D*x, 1));
    constraints = [cvx.norm(y-x,2)<=sigma*np.sqrt(n)]
    prob = cvx.Problem(obj, constraints)
    prob.solve(solver=cvx.ECOS,verbose=False)

    # Check for error.
    if prob.status != cvx.OPTIMAL:
        raise Exception("Solver did not converge!")
        return y
return np.asarray(x.value).flatten()

def merge_patch(rlts, dims, patch_height, patch_width, patch_ref_mat, temp_cor=0.5):
    ########## combine all the components together and find all the boundary components ################
    b_zero = np.zeros((dims[0], dims[1], 1));
    for kk in range(np.prod(patch_ref_mat.shape)):
        ii = np.where(patch_ref_mat == kk)[0][0];
        jj = np.where(patch_ref_mat == kk)[1][0];
        pos = np.where(patch_ref_mat==kk);
        up=pos[0][0]*patch_height;
        down=min(up+patch_height, dims[0]);
        left=pos[1][0]*patch_width;
        right=min(left+patch_width, dims[1]);
        a_temp = rlts[kk][()]["a_fin"].reshape(down-up, right-left, -1, order="F");
        b_temp = rlts[kk][()]["b_fin"].reshape(down-up, right-left, -1, order="F");
        c_temp = rlts[kk][()]["c_fin"];
        a_zero = np.zeros((dims[0], dims[1], c_temp.shape[1]));
        b_zero[up:down,left:right,:] = b_temp;
        a_zero[up:down,left:right,:] = a_temp;
        a_zero = a_zero.reshape(np.prod(dims[:-1]),-1,order="F");
        boundary_comp_temp = np.array([]);
        if ii > 0:
            print("a")
            boundary_comp_temp = np.hstack((boundary_comp_temp, np.where(a_temp[0,:,:].sum(axis=0) > 0)[0]));
        if ii < patch_ref_mat.shape[0] - 1:
            print("b")
            boundary_comp_temp = np.hstack((boundary_comp_temp, np.where(a_temp[down-up-1,:,:].sum(axis=0) > 0)[0]));
        if jj > 0:
            print("c")
            boundary_comp_temp = np.hstack((boundary_comp_temp, np.where(a_temp[:,0,:].sum(axis=0) > 0)[0]));
        if jj < patch_ref_mat.shape[1] - 1:
            print("d")
            boundary_comp_temp = np.hstack((boundary_comp_temp, np.where(a_temp[:,right-left-1,:].sum(axis=0) > 0)[0]));
        boundary_comp_temp = np.unique(boundary_comp_temp);
        if kk == 0:
            a_total = a_zero.copy();
            c_total = c_temp.copy();
            boundary_comp = boundary_comp_temp.copy();
        else:
            boundary_comp_temp = boundary_comp_temp + c_total.shape[1];
            a_total = np.hstack((a_total, a_zero));
            c_total = np.hstack((c_total, c_temp));
            boundary_comp = np.hstack((boundary_comp, boundary_comp_temp));
    boundary_comp = np.asarray(boundary_comp,dtype="int");
    b_zero = b_zero.reshape(np.prod(dims[:-1]),-1,order="F");
    ############## find those boundary components with high temporal correlation #######################
pos_temp = np.where(np.triu(np.corrcoef(c_total[:,boundary_comp].T) > temp_cor, 1));

############## find those boundary components with high temporal correlation and spatially adjacent #######################
s = np.ones([3,3]);
connect_comps0 = [];
connect_comps1 = [];
for jj in range(len(pos_temp[0])):
    temp = a_total[:,[boundary_comp[pos_temp[0][jj]],boundary_comp[pos_temp[1][jj]]]].sum(axis=1).reshape(dims[:-1],order="F");
    _, num_features = scipy.ndimage.measurements.label(temp,structure=s);
    if num_features==1:
        connect_comps0.append(boundary_comp[pos_temp[0][jj]]);
        connect_comps1.append(boundary_comp[pos_temp[1][jj]]);
    if len(connect_comps0) > 0:
        G = nx.Graph();
        G.add_edges_from(list(zip(connect_comps0, connect_comps1)))
        comps=list(nx.connected_components(G))
        
        print("merge" + str(comps));
        merge_idx = np.unique(np.concatenate([connect_comps0, connect_comps1],axis=0));
        a_pri = np.delete(a_total, merge_idx, axis=1);
        c_pri = np.delete(c_total, merge_idx, axis=1);
        for comp in comps:
            comp=list(comp);
            a_zero = np.zeros([a_total.shape[0],1]);
            a_temp = a_total[:,comp];
            mask_temp = np.where(a_temp.sum(axis=1,keepdims=True) > 0)[0];
            a_temp = a_temp[mask_temp,:];
            y_temp = np.matmul(a_temp, c_total[:,comp].T);
            a_temp = a_temp.mean(axis=1,keepdims=True);
            c_temp = c_total[:,comp].mean(axis=1,keepdims=True);
            model = NMF(n_components=1, init='custom')
            a_temp = model.fit_transform(y_temp,W=a_temp, H = (c_temp.T));
            c_temp = (model.components_.T)*np.sqrt((a_temp**2).sum(axis=0));
            a_temp = a_temp/np.sqrt((a_temp**2).sum(axis=0));
            a_zero[mask_temp] = a_temp;
            
            a_pri = np.hstack((a_pri,a_zero));
            c_pri = np.hstack((c_pri,c_temp));
        a_total = a_pri.copy();
        c_total = c_pri.copy();
    else:
        print("no merge!")
        a_pri = a_total.copy();
        c_pri = c_total.copy();
    brightness = np.zeros(a_pri.shape[1]);
    for ii in range(a_total.shape[1]):
        v_max = a_total[:,ii].max();
        u_max = c_total[:,ii].max();
        brightness[ii] = u_max * v_max;
    brightness_rank = a_total.shape[1] - ss.rankdata(brightness,method="ordinal");

for ii in range(a_pri.shape[1]):
    a_pri[:,ii] = a_total[:,np.where(brightness_rank==ii)[0][0]];
    c_pri[:,ii] = c_total[:,np.where(brightness_rank==ii)[0][0]];
    return {'a_pri':a_pri, 'c_pri':c_pri, 'b_pri':b_zero}



