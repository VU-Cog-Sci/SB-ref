
# make scatter plot with size and ecc combined

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


# define participant number and open json parameter file
if len(sys.argv)<2:	
    raise NameError('Please add subject number (ex:01) '	
                    'as 1st argument in the command line!')	

else:	
    sj = str(sys.argv[1]).zfill(2) #fill subject number with 0 in case user forgets	

    with open('analysis_params.json','r') as json_file:	
            analysis_params = json.load(json_file)	


# define paths
figure_out = os.path.join('/home/shared/2018/visual/SB-prep/SB-ref/derivatives/figures','sub-{sj}'.format(sj=sj))

if not os.path.exists(figure_out): # check if path to save figures exists
    os.makedirs(figure_out) 

with_smooth = 'True'#analysis_params['with_smooth']

# load prf estimates
if with_smooth=='True':    
    median_path = os.path.join(analysis_params['pRF_outdir'],'sub-{sj}'.format(sj=sj),'run-median','smooth%d'%analysis_params['smooth_fwhm'])
else:
    median_path = os.path.join(analysis_params['pRF_outdir'],'sub-{sj}'.format(sj=sj),'run-median')



## PRF ##
if sj=='median':
      
    estimates = median_pRFestimates(analysis_params['pRF_outdir'],with_smooth=bool(strtobool(with_smooth)))
    xx = estimates['x']
    yy = estimates['y']
    rsq = estimates['r2']
    size = estimates['size']
    beta = estimates['betas']
    baseline = estimates['baseline']
    
else:
    estimates_list = [x for x in os.listdir(median_path) if x.endswith('estimates.npz')]
    estimates_list.sort() #sort to make sure pRFs not flipped
  
    estimates = []
    for _,val in enumerate(estimates_list) :
        estimates.append(np.load(os.path.join(median_path, val))) #save both hemisphere estimates in same array

    # concatenate r2 and parameteres, to later visualize whole brain
    rsq = np.concatenate((estimates[0]['r2'],estimates[1]['r2']))

    xx = np.concatenate((estimates[0]['x'],estimates[1]['x']))
    yy = np.concatenate((estimates[0]['y'],estimates[1]['y']))

    size = np.concatenate((estimates[0]['size'],estimates[1]['size']))
    baseline = np.concatenate((estimates[0]['baseline'],estimates[1]['baseline']))
    beta = np.concatenate((estimates[0]['betas'],estimates[1]['betas']))


# now construct polar angle and eccentricity values
# grid fit function gives out [x,y,size,baseline,betas]
rsq_threshold = analysis_params['rsq_threshold']

complex_location = xx + yy * 1j
polar_angle = np.angle(complex_location)
eccentricity = np.abs(complex_location)

# get vertices for subject fsaverage
ROIs = ['V1','V2','sPCS','iPCS']
roi_verts = cortex.get_roi_verts('fsaverage',ROIs) # roi_verts is a dictionary


# plot
f, s = plt.subplots(len(ROIs), 2, figsize=(24,48))
for idx,roi in enumerate(ROIs):
    # get datapoints for RF only belonging to roi
    new_rsq = rsq[roi_verts[roi]]

    new_xx = xx[roi_verts[roi]]
    new_yy = yy[roi_verts[roi]]
    new_size = size[roi_verts[roi]]
    
    new_ecc = eccentricity[roi_verts[roi]]
    new_polar_angle = polar_angle[roi_verts[roi]]
    
    # do scatter plot, with RF positions and alpha scaled by rsq
    rgba_colors = np.zeros((new_xx.shape[0], 4))
    rgba_colors[:,0] = 0.8 # make red
    rgba_colors[:,3] = new_rsq/5 #12# the rsq is alpha

    edgecolors = np.zeros((new_xx.shape[0], 4))
    edgecolors[:,:3] = 1#0.8 # make gray
    edgecolors[:,3] = new_rsq/10 #12# the rsq is alpha
    
    s[idx][0].scatter(new_xx, new_yy, s=(4*np.pi*new_size)**2, color=rgba_colors, edgecolors=edgecolors, linewidths=2) # this size is made up and depends on dpi - beware.
    s[idx][0].set_title('%s pRFs in visual field'%roi)
    s[idx][0].set_xlim([-analysis_params["max_eccen"],analysis_params["max_eccen"]])
    s[idx][0].set_ylim([-analysis_params["max_eccen"],analysis_params["max_eccen"]])
    s[idx][0].axvline(0, -15, 15, c='k', lw=0.25)
    s[idx][0].axhline(0, -15, 15, c='k', lw=0.25)
    s[idx][0].set_xlabel('horizontal space [dva]')
    s[idx][0].set_ylabel('vertical space [dva]')
    s[idx][1].set_title('%s pRF size vs eccentricity'%roi)
    s[idx][1].set_xlim([0,15])
    s[idx][1].set_ylim([0,analysis_params["max_size"]])
    s[idx][1].set_xlabel('pRF eccentricity [dva]')
    s[idx][1].set_ylabel('pRF size [dva]')
    s[idx][1].scatter(new_ecc, new_size, color=rgba_colors, edgecolors=edgecolors, linewidths=2);  # this size is made up - beware.
    
f.savefig(os.path.join(figure_out,'RF_scatter_allROIs.png'), dpi=100)

# kernel density plot
for idx,roi in enumerate(ROIs):
    # get datapoints for RF only belonging to roi
    new_rsq = rsq[roi_verts[roi]]

    new_xx = xx[roi_verts[roi]]
    new_yy = yy[roi_verts[roi]]
    new_size = size[roi_verts[roi]]
    
    new_ecc = eccentricity[roi_verts[roi]]
    new_polar_angle = polar_angle[roi_verts[roi]]
    
    g = sns.jointplot(x=new_ecc, y=new_size, kind="kde", color="r",xlim=(0,15),ylim=(0,analysis_params["max_size"]))
    g.set_axis_labels("%s pRF eccentricity [dva]"%roi, " %s pRF size [dva]"%roi)
    
    g.savefig(os.path.join(figure_out,"%s_pRF_size_eccentricity.png"%roi))


    


