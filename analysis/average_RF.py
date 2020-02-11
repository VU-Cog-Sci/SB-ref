# change way of getting median estimates
# (now I'm giving all subs same weight for all voxels)
# so trying to get a more precise account of RF

import os, json
import sys, glob
import numpy as np

from utils import *

import matplotlib.pyplot as plt
from nilearn import surface
from distutils.util import strtobool

import cortex
import seaborn as sns
import pandas as pd


# requires pfpy be installed - preferably with python setup.py develop
from prfpy.rf import *
from prfpy.timecourse import *
from prfpy.stimulus import PRFStimulus2D
from prfpy.grid import Iso2DGaussianGridder,CSS_Iso2DGaussianGridder
from prfpy.fit import Iso2DGaussianFitter, CSS_Iso2DGaussianFitter

from popeye import utilities 

import matplotlib.gridspec as gridspec
import scipy

import matplotlib.patches as patches
from statsmodels.stats import weightstats

import random

from joblib import Parallel, delayed


sj = 'median'

with open('analysis_params.json','r') as json_file: 
            analysis_params = json.load(json_file)  


# used smoothed data (or not) for plots
with_smooth = True

# fit model to use (gauss or css)
fit_model = 'css' #analysis_params["fit_model"]
# if using estimates from iterative fit
iterative_fit = True #True

# total number of chunks that were fitted (per hemi)
total_chunks = analysis_params['total_chunks']


all_estimates = append_pRFestimates(analysis_params['pRF_outdir'],
                          model=fit_model,iterative=iterative_fit,exclude_subs=['sub-07'],total_chunks=total_chunks)


# load all
rsq = np.array(all_estimates['r2'])
ns = np.array(all_estimates['ns'])
betas = np.array(all_estimates['betas'])
baseline = np.array(all_estimates['baseline'])
xx = np.array(all_estimates['x'])
yy = np.array(all_estimates['y'])
size = np.array(all_estimates['size'])


# set limits for xx and yy, forcing it to be within the screen boundaries

vert_lim_dva = (analysis_params['screenRes'][-1]/2) * dva_per_pix(analysis_params['screen_width'],analysis_params['screen_distance'],analysis_params['screenRes'][-1])
hor_lim_dva = (analysis_params['screenRes'][0]/2) * dva_per_pix(analysis_params['screen_width'],analysis_params['screen_distance'],analysis_params['screenRes'][-1])

# make new variables that are masked (center of RF within screen limits and only positive pRFs)
# also max size of RF is 10 dva
print('masking estimates array to be within screen and only show positive RF')

masked_rsq = [] # only need RSQ to be masked, for later plots

for i in range(len(all_estimates['subs'])):
    new_estimates = mask_estimates(xx[i],yy[i],size[i],betas[i],baseline[i],rsq[i],vert_lim_dva,hor_lim_dva,ns=ns[i])
    
    masked_rsq.append(new_estimates['rsq'])


##### load prfpy classes to get single prediction #############

# create/load design matrix
png_path = analysis_params['imgs_dir']
png_filename = [os.path.join(png_path, png) for png in os.listdir(png_path)]
png_filename.sort()

dm_filename = os.path.join(os.getcwd(), 'prf_dm_square.npy')

if not os.path.exists(dm_filename):  # if not exists
    screenshot2DM(png_filename, 0.1,
                  analysis_params['screenRes'], dm_filename,dm_shape = 'square')  # create it
    print('computed %s' % (dm_filename))

else:
    print('loading %s' % dm_filename)

prf_dm = np.load(dm_filename)
prf_dm = prf_dm.T # then it'll be (x, y, t)

# change DM to see if fit is better like that
# do new one which is average of every 2 TRs

prf_dm = shift_DM(prf_dm)
prf_dm = prf_dm[:,:,analysis_params['crop_pRF_TR']:] # crop DM because functional data also cropped now

# define model params
TR = analysis_params["TR"]
hrf = utilities.spm_hrf(0,TR)

# make stimulus object, which takes an input design matrix and sets up its real-world dimensions
prf_stim = PRFStimulus2D(screen_size_cm=analysis_params["screen_width"], 
                         screen_distance_cm=analysis_params["screen_distance"], 
                         design_matrix=prf_dm, 
                         TR=TR)


# divide and save chunks in folder, avoids memory issues
out_dir = os.path.join(analysis_params['derivatives'],'average_RF')
if not os.path.exists(out_dir):  # check if path exists
    os.makedirs(out_dir)

chunk_num = find_zero_remainder(np.array(masked_rsq).shape[-1],100)[-1] #94 chunks
num_vox_chunk = int(np.array(masked_rsq).shape[-1]/chunk_num) # total number of vertices per chunks

print('computing weighted RF') 
for w in range(chunk_num):

    filename = 'average_RF_chunk-%s.npz'%str(w).zfill(2)

    if not os.path.isfile(os.path.join(out_dir,filename)): # if file doesnt exist

      avg_RF = Parallel(n_jobs=16)(delayed(combined_rf)(prf_stim,np.array(xx)[:,i],np.array(yy)[:,i],
                           np.array(size)[:,i],np.array(rsq)[:,i],np.array(ns)[:,i]) 
                         for i in range(num_vox_chunk*w,num_vox_chunk*(w+1)))
      np.savez(os.path.join(out_dir,filename), avg_RF = avg_RF)

    if with_smooth==True:

      filename_sm = 'average_RF_smooth_chunk-%s.npz'%str(w).zfill(2)

      if not os.path.isfile(os.path.join(out_dir,filename_sm)): # if doesnt exist

        RF = np.load(os.path.join(out_dir,filename)) # load
        RF = RF['avg_RF']
        # smooth the RF
        avg_RF = Parallel(n_jobs=16)(delayed(smooth_RF)(RF[i,:,:],sigma=20) for i in range(RF.shape[0]))
        np.savez(os.path.join(out_dir,filename_sm), avg_RF = avg_RF)


# compute polar angle value  for those RF
print('computing polar angle values')    
for w in range(chunk_num):

    if not os.path.isfile(os.path.join(out_dir,'average_PA_chunk-%s.npz'%str(w).zfill(2))):

      RF_filename = 'average_RF_chunk-%s.npz'%str(w).zfill(2)
      loaded_RF = np.load(os.path.join(out_dir,RF_filename))
      loaded_RF = loaded_RF['avg_RF']

      avg_pa = Parallel(n_jobs=16)(delayed(max_RF2PA)(loaded_RF[i,:,:]) for i in range(loaded_RF.shape[0]))
      np.savez(os.path.join(out_dir,'average_PA_chunk-%s.npz'%str(w).zfill(2)), avg_pa = avg_pa)

      # repetitive but ok for now
      RF_filename = 'average_RF_smooth_chunk-%s.npz'%str(w).zfill(2)
      loaded_RF = np.load(os.path.join(out_dir,RF_filename))
      loaded_RF = loaded_RF['avg_RF']

      avg_pa = Parallel(n_jobs=16)(delayed(max_RF2PA)(loaded_RF[i,:,:]) for i in range(loaded_RF.shape[0]))
      np.savez(os.path.join(out_dir,'average_PA_smooth_chunk-%s.npz'%str(w).zfill(2)), avg_pa = avg_pa)


# load all polar angles and make flatmap
if with_smooth:
  all_pa_str = [x for x in os.listdir(out_dir) if 'PA' in x and 'smooth' in x]; all_pa_str.sort()  # load the PA from not smoothed RF (I think it doesn't make a difference, because I'm using COM)
else:
  all_pa_str = [x for x in os.listdir(out_dir) if 'PA' in x and 'smooth' not in x]; all_pa_str.sort()  # load the PA from not smoothed RF (I think it doesn't make a difference, because I'm using COM)


all_pa = []
for w in range(chunk_num):
    if 'chunk-%s.npz'%str(w).zfill(2) in all_pa_str[w]:
        arr = np.load(os.path.join(out_dir,all_pa_str[w]))
        all_pa.append(arr['avg_pa'])
        
    else:
        print('Chunk %d missing'%w)


# get median rsq array, to mask flatmap (PA array does not account for screen limits etc)
masked_rsq = np.nanmedian(np.array(masked_rsq),axis=0)

# now construct polar angle and eccentricity values
rsq_threshold = 0.14 #0.125 #analysis_params['rsq_threshold']

masked_polar_angle = np.array(np.ravel(all_pa))

# normalize polar angles to have values in circle between 0 and 1
masked_polar_ang_norm = (masked_polar_angle + np.pi) / (np.pi * 2.0)

# use "resto da divisão" so that 1 == 0 (because they overlapp in circle)
# why have an offset?
angle_offset = 0.85#0.1
masked_polar_ang_norm = np.fmod(masked_polar_ang_norm+angle_offset, 1.0)

# convert angles to colors, using correlations as weights
hsv = np.zeros(list(masked_polar_ang_norm.shape) + [3])
hsv[..., 0] = masked_polar_ang_norm # different hue value for each angle
hsv[..., 1] = (masked_rsq > rsq_threshold).astype(float)#  np.ones_like(rsq) # saturation weighted by rsq
hsv[..., 2] = (masked_rsq > rsq_threshold).astype(float) # value weighted by rsq

# convert hsv values of np array to rgb values (values assumed to be in range [0, 1])
rgb = colors.hsv_to_rgb(hsv)

# define alpha channel - which specifies the opacity for a color
# define mask for alpha, to be all values where rsq below threshold or nan 
alpha_mask = np.array([True if val<= rsq_threshold or np.isnan(val) else False for _,val in enumerate(masked_rsq)])

# create alpha array weighted by rsq values
alpha = np.sqrt(masked_rsq.copy())#np.ones(alpha_mask.shape)
alpha[alpha_mask] = np.nan

# create alpha array with nan = transparent = values with rsq below thresh and 1 = opaque = values above thresh
alpha_ones = np.ones(alpha_mask.shape)
alpha_ones[alpha_mask] = np.nan


images = {}

images['polar'] = cortex.VertexRGB(rgb[..., 0], 
                                 rgb[..., 1], 
                                 rgb[..., 2], 
                                 subject='fsaverage_gross', alpha=alpha)
#cortex.quickshow(images['polar'],with_curvature=True,with_sulci=True,with_colorbar=False)

filename = os.path.join(out_dir,'flatmap_space-fsaverage_rsq-%0.2f_type-polar_angle.svg' %rsq_threshold)
if with_smooth: filename = filename.replace('polar_angle.svg','polar_angle_smooth_%d.svg'%analysis_params['smooth_fwhm'])

print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['polar'], recache=True,with_colorbar=False,with_curvature=True,with_sulci=True)


images['polar_noalpha'] = cortex.VertexRGB(rgb[..., 0], 
                                 rgb[..., 1], 
                                 rgb[..., 2], 
                                 subject='fsaverage_gross', alpha=alpha_ones)
#cortex.quickshow(images['polar_noalpha'],with_curvature=True,with_sulci=True,with_colorbar=False)

filename = os.path.join(out_dir,'flatmap_space-fsaverage_rsq-%0.2f_type-polar_angle_noalpha.svg' %rsq_threshold)
if with_smooth: filename = filename.replace('_noalpha.svg','_noalpha_smooth_%d.svg'%analysis_params['smooth_fwhm'])

print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['polar_noalpha'], recache=True,with_colorbar=False,with_curvature=True,with_sulci=True)






















