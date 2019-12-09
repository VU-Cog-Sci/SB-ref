# make prf and soma rsq maps
# combined - ideally to see which regions are motor and which are visual and where these overlap
# load estimates that were obtained previously in other scripts, so need to run individual task scripts first

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

import scipy
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as matcm

# define participant number and open json parameter file
if len(sys.argv)<2: 
    raise NameError('Please add subject number (ex:01) '    
                    'as 1st argument in the command line!') 

else:   
    sj = str(sys.argv[1]).zfill(2) #fill subject number with 0 in case user forgets 

    with open('analysis_params.json','r') as json_file: 
            analysis_params = json.load(json_file)  


# define paths
with_smooth = 'True'#'False'#analysis_params['with_smooth']

if with_smooth=='True':
    figure_out = os.path.join(analysis_params['derivatives'],'figures','multimodal','sub-{sj}'.format(sj=sj),'smooth%d'%analysis_params['smooth_fwhm'])
    # dir to get soma contrasts
    soma_dir = os.path.join(analysis_params['soma_outdir'],'new_fits','sub-{sj}'.format(sj=sj),'smooth%d'%analysis_params['smooth_fwhm'])
    # dir to get prf (masked) estimates
    prf_dir = os.path.join(analysis_params['derivatives'],'figures','prf','shift_crop','sub-{sj}'.format(sj=sj),'smooth%d'%analysis_params['smooth_fwhm'])
else:
    figure_out = os.path.join(analysis_params['derivatives'],'figures','multimodal','sub-{sj}'.format(sj=sj))
    # dir to get soma contrasts
    soma_dir = os.path.join(analysis_params['soma_outdir'],'new_fits','sub-{sj}'.format(sj=sj)) 
    # dir to get prf estimates
    prf_dir = os.path.join(analysis_params['derivatives'],'figures','prf','shift_crop','sub-{sj}'.format(sj=sj))


if not os.path.exists(figure_out): # check if path to save figures exists
    os.makedirs(figure_out) 


# load prf masked estimates (positive RF and within screen boundaries)
prf_estimates = np.load(os.path.join(prf_dir,'masked_estimates.npz'),allow_pickle=True)
# load soma estimates
soma_estimates = np.load(os.path.join(soma_dir,[x for _,x in enumerate(os.listdir(soma_dir)) if x.endswith('estimates.npz')][0]),allow_pickle=True)

# save rsq for different models
rsq_visual = prf_estimates['masked_rsq'].astype(float)
rsq_motor = soma_estimates['r2'].astype(float)

# normalize RSQ 
rsq_visual_norm = normalize(rsq_visual) 
rsq_motor_norm = normalize(rsq_motor)

# and then get R values
r_visual = np.sqrt(rsq_visual_norm)
r_motor = np.sqrt(rsq_motor_norm)

# create costume colormp red blue
n_bins = 4
col2D_name = os.path.splitext(os.path.split(make_2D_colormap(rgb_color='101',bins=n_bins))[-1])[0]
print('created costum colormap %s'%col2D_name)

###################
print('making histograms')
# look at histogram of rsq values for the different task, just to get a sense of the distribuition
fig, axis = plt.subplots(1,2,figsize=(15,7.5),dpi=100)

sns.distplot(rsq_visual,bins=10,
            color='r', ax=axis[0])
axis[0].set_xlabel('R-squared',fontsize=14)
axis[0].set_title('Histogram of rsq values from prf fit')

sns.distplot(rsq_motor,bins=10,
            color='b', ax=axis[1])
axis[1].set_xlabel('R-squared',fontsize=14)
axis[1].set_title('Histogram of rsq values from soma fit')

fig.savefig(os.path.join(figure_out,'histogram_combined_rsq.svg'), dpi=100,bbox_inches = 'tight')


# look at histogram of rsq values for the different task, just to get a sense of the distribuition
fig, axis = plt.subplots(1,2,figsize=(15,7.5),dpi=100)

sns.distplot(rsq_visual_norm,bins=10,
            color='r', ax=axis[0])
axis[0].set_xlabel('R-squared',fontsize=14)
axis[0].set_title('Histogram of normalized rsq values from prf fit')

sns.distplot(rsq_motor_norm,bins=10,
            color='b', ax=axis[1])
axis[1].set_xlabel('R-squared',fontsize=14)
axis[1].set_title('Histogram of normalized rsq values from soma fit')

fig.savefig(os.path.join(figure_out,'histogram_combined_rsq_normalized.svg'), dpi=100,bbox_inches = 'tight')


fig, axis = plt.subplots(1,2,figsize=(15,7.5),dpi=100)

sns.distplot(r_visual,bins=10,
            color='r', ax=axis[0])
axis[0].set_xlabel('R',fontsize=14)
axis[0].set_xlim(0,)

axis[0].set_title('Histogram of R values from visual fit')

sns.distplot(r_motor,bins=10,
            color='b', ax=axis[1])
axis[1].set_xlabel('R',fontsize=14)
axis[0].set_xlim(0,)
axis[1].set_title('Histogram of R values from soma fit')

fig.savefig(os.path.join(figure_out,'histogram_combined_R.svg'), dpi=100,bbox_inches = 'tight')

# get threshold values that defines 4th quantile (top 25% of distribution)

quantile = .75 # set quantile value to plot upper part of the distribution of normalized R
vis_quant = np.nanquantile(r_visual, quantile)
mot_quant = np.nanquantile(r_motor, quantile)

fig, axis = plt.subplots(1,2,figsize=(15,7.5),dpi=100)

sns.distplot(r_visual,bins=10,
            color='r', ax=axis[0])
axis[0].set_xlabel('R',fontsize=14)
axis[0].set_xlim(0,)
axis[0].axvline(x=vis_quant,c='k',linestyle='--')

axis[0].set_title('Histogram of R values from visual fit')

sns.distplot(r_motor,bins=10,
            color='b', ax=axis[1])
axis[1].set_xlabel('R',fontsize=14)
axis[0].set_xlim(0,)
axis[1].axvline(x=mot_quant,c='k',linestyle='--')
axis[1].set_title('Histogram of R values from soma fit')

fig.savefig(os.path.join(figure_out,'histogram_combined_R_%.2f-quantile.svg'%quantile), dpi=100,bbox_inches = 'tight')

# make flatmaps of the above distributions ##########
print('making flatmaps')

images = {}

# make and save rsq flatmaps for each task

images['rsq_visual_norm'] = cortex.Vertex(rsq_visual_norm,'fsaverage_gross',
                           vmin=0, vmax=1,
                           cmap='Reds')
cortex.quickshow(images['rsq_visual_norm'],with_curvature=True,with_sulci=True)
filename = os.path.join(figure_out,'flatmap_space-fsaverage_type-rsquared-normalized_visual.svg')
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['rsq_visual_norm'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True)


images['rsq_motor_norm'] = cortex.Vertex(rsq_motor_norm,'fsaverage_gross',
                           vmin=0, vmax=1,
                           cmap='Blues')
cortex.quickshow(images['rsq_motor_norm'],with_curvature=True,with_sulci=True)
filename = os.path.join(figure_out,'flatmap_space-fsaverage_type-rsquared-normalized_motor.svg')
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['rsq_motor_norm'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True)

images['rsq_combined'] = cortex.Vertex2D(rsq_visual_norm,rsq_motor_norm, 
                            subject='fsaverage_gross',
                            vmin=0, vmax=1,
                            vmin2=0,vmax2=1,
                            cmap=col2D_name)#'PU_RdBu_covar')
cortex.quickshow(images['rsq_combined'],with_curvature=True,with_sulci=True)
filename = os.path.join(figure_out,'flatmap_space-fsaverage_type-rsquared-normalized_combined_bins-%d.svg'%n_bins)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['rsq_combined'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True)



# correlation coeficient for each task
images['R_visual'] = cortex.Vertex(r_visual,'fsaverage_gross',
                           vmin=0, vmax=1,
                           cmap='Reds')
cortex.quickshow(images['R_visual'],with_curvature=True,with_sulci=True)
filename = os.path.join(figure_out,'flatmap_space-fsaverage_type-R_visual.svg')
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['R_visual'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True)


images['R_motor'] = cortex.Vertex(r_motor,'fsaverage_gross',
                           vmin=0, vmax=1,
                           cmap='Blues')
cortex.quickshow(images['R_motor'],with_curvature=True,with_sulci=True)
filename = os.path.join(figure_out,'flatmap_space-fsaverage_type-R_motor.svg')
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['R_motor'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True)

images['R_combined'] = cortex.Vertex2D(r_visual, r_motor,
                            subject='fsaverage_gross',
                            vmin=0, vmax=1,
                            vmin2=0,vmax2=1,
                            cmap=col2D_name)#'YeBu_covar_costum')#'PU_RdBu_covar')
cortex.quickshow(images['R_combined'],with_curvature=True,with_sulci=True)
filename = os.path.join(figure_out,'flatmap_space-fsaverage_type-R_combined_bins-%d.svg'%n_bins)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['R_combined'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True)


# mask according to upper quantile threshold

rsq_threshold_visual = vis_quant**2
rsq_threshold_motor = mot_quant**2

alpha_mask_visual = np.array([True if val< rsq_threshold_visual or np.isnan(val) else False for _,val in enumerate(rsq_visual_norm)])
alpha_mask_motor = np.array([True if val< rsq_threshold_motor or np.isnan(val) else False for _,val in enumerate(rsq_motor_norm)])

rsq_visual_alpha = rsq_visual_norm.copy()
rsq_visual_alpha[alpha_mask_visual] = 0 #np.nan

images['rsq_visual_norm_masked'] = cortex.Vertex(rsq_visual_alpha, 'fsaverage_gross',
                           vmin=0, vmax=1,
                           cmap='Reds')
cortex.quickshow(images['rsq_visual_norm_masked'],with_curvature=True,with_sulci=True)
filename = os.path.join(figure_out,'flatmap_space-fsaverage_type-rsquared_visual_masked_rsq-%.2f.svg'%rsq_threshold_visual)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['rsq_visual_norm_masked'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True)


rsq_motor_alpha = rsq_motor_norm.copy()
rsq_motor_alpha[alpha_mask_motor] = 0 #np.nan

images['rsq_motor_norm_masked'] = cortex.Vertex(rsq_motor_alpha, 'fsaverage_gross',
                           vmin=0, vmax=1,
                           cmap='Blues')
cortex.quickshow(images['rsq_motor_norm_masked'],with_curvature=True,with_sulci=True)
filename = os.path.join(figure_out,'flatmap_space-fsaverage_type-rsquared_motor_masked_rsq-%.2f.svg'%rsq_threshold_motor)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['rsq_motor_norm_masked'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True)

images['rsq_combined_masked'] = cortex.Vertex2D(rsq_visual_alpha, rsq_motor_alpha,
                            subject='fsaverage_gross',
                            vmin=0, vmax=1,
                            vmin2=0,vmax2=1,
                            cmap=col2D_name)#'PU_RdBu_covar')
cortex.quickshow(images['rsq_combined_masked'],with_curvature=True,with_sulci=True)
filename = os.path.join(figure_out,'flatmap_space-fsaverage_type-rsquared_combined_masked_rsqvis-%.2f_rsqmot-%.2f_bins-%d.svg'%(rsq_threshold_visual,rsq_threshold_motor,n_bins))
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['rsq_combined_masked'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True)

# also mask R plots, with quantile threshold

quant_R_mask_visual = np.array([True if val< vis_quant or np.isnan(val) else False for _,val in enumerate(r_visual)])
quant_R_mask_motor = np.array([True if val< mot_quant or np.isnan(val) else False for _,val in enumerate(r_motor)])

R_visual_alpha = r_visual.copy()
R_visual_alpha[quant_R_mask_visual] = 0 #np.nan

images['R_visual_alpha'] = cortex.Vertex(R_visual_alpha, 'fsaverage_gross',
                           vmin=0, vmax=1,
                           cmap='Reds')
cortex.quickshow(images['R_visual_alpha'],with_curvature=True,with_sulci=True)
filename = os.path.join(figure_out,'flatmap_space-fsaverage_type-R-normalized_visual_masked_R-%.2f.svg'%vis_quant)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['R_visual_alpha'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True)

R_motor_alpha = r_motor.copy()
R_motor_alpha[quant_R_mask_motor] = 0 #np.nan

images['R_motor_alpha'] = cortex.Vertex(R_motor_alpha, 'fsaverage_gross',
                           vmin=0, vmax=1,
                           cmap='Blues')
cortex.quickshow(images['R_motor_alpha'],with_curvature=True,with_sulci=True)
filename = os.path.join(figure_out,'flatmap_space-fsaverage_type-R-normalized_motor_masked_R-%.2f.svg'%mot_quant)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['R_motor_alpha'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True)

images['R_combined_masked'] = cortex.Vertex2D(R_visual_alpha,R_motor_alpha, 
                            subject='fsaverage_gross',
                            vmin=0, vmax=1,
                            vmin2=0,vmax2=1,
                            cmap=col2D_name)#'YeBu_covar_costum')#'PU_RdBu_covar')
cortex.quickshow(images['R_combined_masked'],with_curvature=True,with_sulci=True)
filename = os.path.join(figure_out,'flatmap_space-fsaverage_type-R-normalized_combined_masked_Rvisual-%.2f_Rmotor-%.2f_bins-%d.svg'%(vis_quant,mot_quant,n_bins))
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['R_combined_masked'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True)

print('Done!')




