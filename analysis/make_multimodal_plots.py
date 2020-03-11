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

# path to save figure
figure_out = os.path.join(analysis_params['derivatives'],'figures','multimodal','sub-{sj}'.format(sj=sj))
# dir to get soma contrasts
soma_dir = os.path.join(analysis_params['soma_outdir'],'new_fits','sub-{sj}'.format(sj=sj)) 

if with_smooth=='True':
    figure_out = os.path.join(figure_out,'smooth%d'%analysis_params['smooth_fwhm'])
    soma_dir = os.path.join(soma_dir,'smooth%d'%analysis_params['smooth_fwhm'])

if not os.path.exists(figure_out): # check if path to save figures exists
    os.makedirs(figure_out) 

### PRF PARAMS #######
# fit model to use (gauss or css)
fit_model = 'css' #analysis_params["fit_model"]

# if using estimates from iterative fit
iterative_fit = True #True

# total number of chunks that were fitted (per hemi)
total_chunks = analysis_params['total_chunks']
    
##

# filename for visual rsq
rsq_visual_filename = os.path.join(figure_out,'rsq_pRF_fitmodel-%s_itertivefit-%s.npy'%(fit_model,str(iterative_fit)))

if not os.path.isfile(rsq_visual_filename):
    # make list with subjects to append and use (or not)
    if sj == 'median':
        excl_subs = ['sub-07']
    else:
        all_subs = ['01','02','03','04','05','07','08','09','11','12','13']
        excl_subs = ['sub-'+name for _,name in enumerate(all_subs) if name!=sj]
        
    # first append estimates (if median) or load if single sub
    estimates = append_pRFestimates(analysis_params['pRF_outdir'],
                                        model=fit_model,iterative=iterative_fit,exclude_subs=excl_subs,total_chunks=total_chunks)

    print('appended estimates for %s excluded %s'%(str(estimates['subs']),str(estimates['exclude_subs'])))

    rsq_visual = np.array(estimates['r2'])
    
    # now mask them according to screen dimensions
    masked_rsq = []

    for w in range(rsq_visual.shape[0]): # loop once if one subject, or for all subjects when sj 'all'

        subject = estimates['subs'][w]

        # set limits for xx and yy, forcing it to be within the screen boundaries
        if subject in ['sub-02','sub-11','sub-12','sub-13']: # linux computer has different res
            vert_lim_dva = (analysis_params['screenRes_HD'][-1]/2) * dva_per_pix(analysis_params['screen_width'],analysis_params['screen_distance'],analysis_params['screenRes_HD'][0])
            hor_lim_dva = (analysis_params['screenRes_HD'][0]/2) * dva_per_pix(analysis_params['screen_width'],analysis_params['screen_distance'],analysis_params['screenRes_HD'][0])
        else:    
            vert_lim_dva = (analysis_params['screenRes'][-1]/2) * dva_per_pix(analysis_params['screen_width'],analysis_params['screen_distance'],analysis_params['screenRes'][0])
            hor_lim_dva = (analysis_params['screenRes'][0]/2) * dva_per_pix(analysis_params['screen_width'],analysis_params['screen_distance'],analysis_params['screenRes'][0])

        # make new variables that are masked (center of RF within screen limits and only positive pRFs)
        # also max size of RF is 10 dva
        print('masking variables to be within screen and only show positive RF for sub %s'%subject)
        new_estimates = mask_estimates(estimates['x'][w],estimates['y'][w],
                                       estimates['size'][w],estimates['betas'][w],
                                       estimates['baseline'][w],rsq_visual[w],
                                       vert_lim_dva,hor_lim_dva,ns=estimates['ns'][w])

        masked_rsq.append(new_estimates['rsq'])

    # make median and save
    rsq_visual = np.nanmedian(np.array(rsq_visual),axis=0)
    
    if with_smooth == 'True':
        
        header_sub = '01' if sj=='median' else sj # to get header for smoothing
            
        rsq_visual = smooth_nparray(os.path.join(analysis_params['post_fmriprep_outdir'], 'prf', 'sub-{sj}'.format(sj=str(header_sub).zfill(2))),
                           rsq_visual,
                           figure_out,
                           'rsq',
                           rsq_visual_filename,
                           n_TR=141,
                           task='prf',
                           file_extension='_cropped_sg_psc.func.gii',
                           sub_mesh='fsaverage',
                           smooth_fwhm=analysis_params['smooth_fwhm'])
        
    
    np.save(rsq_visual_filename,rsq_visual)
    
else:
    print('loading visual rsq from %s'%rsq_visual_filename)
    rsq_visual = np.load(rsq_visual_filename,allow_pickle=True)


# load soma rsq
print('load %s'%os.path.join(soma_dir,'rsq.npy'))
rsq_motor = np.load(os.path.join(soma_dir,'rsq.npy'))


# do this to replace nans with 0s, so flatmaps look nicer
rsq_visual = np.array([x if np.isnan(x)==False else 0 for _,x in enumerate(rsq_visual)])
rsq_motor = np.array([x if np.isnan(x)==False else 0 for _,x in enumerate(rsq_motor)])

# normalize RSQ 
rsq_visual_norm = normalize(rsq_visual) 
rsq_motor_norm = normalize(rsq_motor)


# create costume colormp red blue
n_bins = 256
col2D_name = os.path.splitext(os.path.split(make_2D_colormap(rgb_color='101',bins=n_bins))[-1])[0]
print('created costum colormap %s'%col2D_name)


# make flatmaps of the above distributions
print('making flatmaps')

images = {}

images['rsq_visual_norm'] = cortex.Vertex(rsq_visual_norm,'fsaverage_gross',
                           vmin=0, vmax=1,
                           cmap='Reds')
#cortex.quickshow(images['rsq_visual_norm'],with_curvature=True,with_sulci=True)
filename = os.path.join(figure_out,'flatmap_space-fsaverage_type-rsquared-normalized_visual.svg')
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['rsq_visual_norm'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True)

images['rsq_motor_norm'] = cortex.Vertex(rsq_motor_norm,'fsaverage_gross',
                           vmin=0, vmax=1,
                           cmap='Blues')
#cortex.quickshow(images['rsq_motor_norm'],with_curvature=True,with_sulci=True)
filename = os.path.join(figure_out,'flatmap_space-fsaverage_type-rsquared-normalized_motor.svg')
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['rsq_motor_norm'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True)


images['rsq_combined'] = cortex.Vertex2D(rsq_visual_norm,rsq_motor_norm, 
                            subject='fsaverage_gross',
                            vmin=0.125, vmax=.25,
                            vmin2=0.2,vmax2=0.7,
                            cmap=col2D_name)#'PU_RdBu_covar')
#cortex.quickshow(images['rsq_combined'],recache=True,with_curvature=True,with_sulci=True,with_roi=False,height=2048)
filename = os.path.join(figure_out,'flatmap_space-fsaverage_type-rsquared-normalized_combined_bins-%d.svg'%n_bins)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['rsq_combined'], recache=True,with_roi=True,with_colorbar=False,with_curvature=True,with_sulci=True,height=2048)


# Plot a flatmap with the data projected onto the surface
# Highlight the curvature and which cutout to be displayed

# Name of a sub-layer of the 'cutouts' layer in overlays.svg file
cutout_name = 'zoom_roi_left'
_ = cortex.quickflat.make_figure(images['rsq_combined'],
                                 with_curvature=True,
                                 with_sulci=True,
                                 with_roi=False,
                                 with_colorbar=False,
                                 cutout=cutout_name,height=2048)

filename = os.path.join(figure_out,cutout_name+'_space-fsaverage_type-rsquared-normalized_combined_bins-%d.svg'%n_bins)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['rsq_combined'], recache=True,with_colorbar=False,
                              cutout=cutout_name,with_curvature=True,with_sulci=True,with_roi=False,height=2048)


# Name of a sub-layer of the 'cutouts' layer in overlays.svg file
cutout_name = 'zoom_roi_right'
_ = cortex.quickflat.make_figure(images['rsq_combined'],
                                 with_curvature=True,
                                 with_sulci=True,
                                 with_roi=False,
                                 with_colorbar=False,
                                 cutout=cutout_name,height=2048)
filename = os.path.join(figure_out,cutout_name+'_space-fsaverage_type-rsquared-normalized_combined_bins-%d.svg'%n_bins)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['rsq_combined'], recache=True,with_colorbar=False,
                              cutout=cutout_name,with_curvature=True,with_sulci=True,with_roi=False,height=2048)








print('Done!')




