import superpixel_analysis as sup
import numpy as np

def spatial_filter_image(Y_new, gHalf=[2,2], sn=None):
    """
    Apply a wiener filter to image Y_new d1 x d2 x T
    """
    if sn is None:
        sn = sup.noise_estimator(Y_new)
    else:
        print('sn given')

    Y_new1 = np.zeros(Y_new.shape);
    d = np.shape(Y_new)
    n_pixels = np.prod(d[:-1])

    k_hats=[]
    for pixel in np.arange(n_pixels):
        if pixel % 1e3==0:
            print('first k pixels %d'%pixel)
        ij = np.unravel_index(pixel,d[:2])
        # Get surrounding area
        ijSig = [[np.maximum(ij[c] - gHalf[c], 0), np.minimum(ij[c] + gHalf[c] + 1, d[c])]
                for c in range(len(ij))]

        Y_curr = np.array(Y_new[[slice(*a) for a in ijSig]].copy(),dtype=np.float32)
        sn_curr = np.array(sn[[slice(*a) for a in ijSig]].copy(),dtype=np.float32)
        #print('shape is %d %d'%(sn_curr.shape))
        cc1 = ij[0]-ijSig[0][0] # center
        cc2 = ij[1]-ijSig[1][0]
        #print('Index {} center {}x {}'.format(k,cc1,cc2))
        neuron_indx = int(np.ravel_multi_index((cc1,cc2),Y_curr.shape[:2],order='F'))
        Y_out, k_hat = spatial_filter_block(Y_curr, sn=sn_curr,
                neuron_indx=neuron_indx)
        Y_new1[ij[0],ij[1],:] = Y_out[cc1,cc2,:];
        k_hats.append(k_hat);
    return Y_new1, k_hats

def spatial_filter_block(data,sn,neuron_indx=None):
    """
    Apply wiener filter to block in data d1 x d2 x T
    """
    data = np.asarray(data)
    dims = data.shape
    mean_ = data.mean(2,keepdims=True)
    data_ = data - mean_

    sn = sn.reshape(np.prod(dims[:2]),order='F')
    D = np.diag(sn**2)
    data_r = data_.reshape((np.prod(dims[:2]),dims[2]),order='F')
    Cy = data_r.dot(data_r.T)/(data_r.shape[1]-1)

    try:
        if neuron_indx is None:
            hat_k = np.linalg.inv(Cy).dot(Cy-D)
        else:
            hat_k = np.linalg.inv(Cy).dot(Cy[neuron_indx,:]-D[neuron_indx,:])
    except np.linalg.linalg.LinAlgError as err:
        print('Singular matrix(?) bye bye')
        return data, []

    if neuron_indx is None:
        y_ = hat_k.dot(data_r)
    else:
        y_ = data_r.copy()
        y_[neuron_indx,:] = hat_k[:,np.newaxis].T.dot(data_r)
    y_hat = y_.reshape(dims[:2]+(dims[2],),order='F')
    y_hat = y_hat + mean_

    return y_hat, hat_k
