import numpy as np
import scipy.signal
from scipy import ndimage
from sklearn import preprocessing
import numpy.matlib


def get_encoded(encoder_model, PI_nor):
    return encoder_model.predict(PI_nor)


def compute_PI(trajectory: np.ndarray, n1: int = 20, n2: int = 2, beta: float = np.pi) -> np.ndarray:
    # the head direction + path integration block from train.py
    # returns PI_nor
    dphi = 180 / n2
    dtheta = 360 / n1
    phis = np.arange(0, 180 + 1, dphi)
    thetas = np.arange(0, 360, dtheta)


    # u_i
    temp_phis = np.repeat(phis, len(thetas))
    temp_thetas = np.matlib.repmat(thetas, 1,len(phis))
    temp_phis = np.transpose(np.asarray(temp_phis))
    pref_dir = np.column_stack((temp_thetas[0,:], temp_phis))

    angle_pref_x = np.sin(np.radians(pref_dir[:,1]))*np.cos(np.radians(pref_dir[:,0]))
    angle_pref_y = np.sin(np.radians(pref_dir[:,1]))*np.sin(np.radians(pref_dir[:,0]))
    angle_pref_z = np.cos(np.radians(pref_dir[:,1]))

    # Z.u_i calculation
    HD = np.zeros((len(pref_dir), trajectory.shape[0]))
    print(len(pref_dir))
    for i in range(trajectory.shape[0]):
        for j in range(len(pref_dir)):
            HD[j,i] = ((trajectory[i,0]-trajectory[0,0])*angle_pref_x[j]) + ((trajectory[i,1]-trajectory[0,1])*angle_pref_y[j]) + ((trajectory[i,2]-trajectory[0,2])*angle_pref_z[j])#incomplete PI


    """#Path Integration"""
    PI = np.transpose(np.cos(beta*(HD)))
    PI_nor = preprocessing.normalize(PI, norm='l2')

    return PI_nor

def firing_rate_map(ot, all_dat, thresh_param = 0.0, res_param = 40, sub_thresh=False):
  res = res_param
  if sub_thresh:
    mean_resp = np.mean(ot)
    std_resp = np.std(ot)
    thresh = mean_resp + (thresh_param * std_resp)
  else:
    thresh = np.max(ot)*thresh_param

  firr = np.nonzero(ot>thresh)
  firtrajectorygrid = all_dat[firr[0], :2]

  x = np.arange(1, 6, 1/res)
  y = np.arange(1, 6, 1/res)


  fx,fy = np.meshgrid(x, y)
  firingmap = np.zeros(fx.shape)
  firingvalue = abs(ot[firr])
  for ii in range(len(firtrajectorygrid)):
    q1 = np.argmin(abs(firtrajectorygrid[ii,0] - fx[1,:]))
    q2 = np.argmin(abs(firtrajectorygrid[ii,1] - fy[:,1]))
    firingmap[q2,q1] = max(firingvalue[ii], firingmap[q2,q1])

  firingmap = firingmap/max(np.max(firingmap),1)
  win_len = 9
  gaussian = matlab_style_gauss2D([win_len, win_len], 3.0)
  spikes_smooth = scipy.signal.convolve2d(gaussian, firingmap)
  rotated_img = ndimage.rotate(spikes_smooth, 1*0)

  rotated_img = rotated_img[int(win_len/2):-int(win_len/2), int(win_len/2):-int(win_len/2)]
  return rotated_img


def matlab_style_gauss2D(shape,sigma):
  """
  2D gaussian mask - should give the same result as MATLAB's
  fspecial('gaussian',[shape],[sigma])
  """
  m,n = [(ss-1.)/2. for ss in shape]
  y,x = np.ogrid[-m:m+1,-n:n+1]
  h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
  h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
  sumh = h.sum()
  if sumh != 0:
    h /= sumh
  return h

def _inf_rate(rate_map, px):
    '''A helper function for information rate.'''
    tmp_map = np.ma.array(rate_map, copy=True)
    tmp_map[np.isnan(tmp_map)] = 0
    avg_rate = np.sum(np.ravel(tmp_map * px))
    
    if avg_rate == 0:
        return (0.0, 0.0)
    else:
        return (np.nansum(np.ravel(tmp_map * np.log2(tmp_map/avg_rate) * px)),
                avg_rate)

def compute_spatial_info(rate_map, trajectory_prob):
    sp_info_rate, avg_rate = _inf_rate(rate_map, trajectory_prob)
    if avg_rate == 0:
        return 0.0
    else:
        return sp_info_rate * 0.1 / avg_rate


def rate_map_analysis(encoded1, pos, reso: int = 8):
    H1, _, _ = np.histogram2d(pos[:,0], pos[:,1], bins=reso*5)
    pos_prob = (H1)/len(pos)
    firing_maps = []
    spatial_info = []
    lim = 2.0

    resp_neurons = encoded1.T
    for i in range(resp_neurons.shape[0]):
        img_dat = firing_rate_map(resp_neurons[i], pos, thresh_param=lim,
                                    res_param = reso)

        # firing_maps.append(img_dat)
        spatial_info.append(compute_spatial_info(img_dat, pos_prob))
        # spatial_info[outs[k]].append(sp_info_rate/avg_rate)
        firing_maps.append(img_dat.tolist())

    return firing_maps, spatial_info