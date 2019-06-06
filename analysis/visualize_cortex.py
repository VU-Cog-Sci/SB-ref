#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Th Jun 6 11:13:11 2019

@author: inesverissimo

Do pRF fit on median run and save outputs
"""

import os, json
import sys, glob
import numpy as np
import cortex
import matplotlib.colors as colors


# define participant number and open json parameter file
if len(sys.argv)<2:	
    raise NameError('Please add subject number (ex:01) '	
                    'as 1st argument in the command line!')	
   	 
else:	
    sj = str(sys.argv[1]).zfill(2) #fill subject number with 0 in case user forgets	

    with open('analysis_params.json','r') as json_file:	
            analysis_params = json.load(json_file)	
    

## PRF ##

# load prf estimates
median_path = os.path.join(analysis_params['pRF_outdir'],'sub-{sj}'.format(sj=sj),'run-median')
estimates_list = [x for x in os.listdir(median_path) if x.endswith('.estimates.npz') ]
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

# normalize polar angles to have values in circle between 0 and 1 
polar_ang_norm = (polar_angle + np.pi) / (np.pi * 2.0)

# use "resto da divisão" so that 1 == 0 (because they overlapp in circle)
# why have an offset?
angle_offset = 0.1
polar_ang_norm = np.fmod(polar_ang_norm+angle_offset, 1.0)

# convert angles to colors, using correlations as weights
hsv = np.zeros(list(polar_ang_norm.shape) + [3])
hsv[..., 0] = polar_ang_norm # different hue value for each angle
hsv[..., 1] = (rsq > rsq_threshold).astype(float)#  np.ones_like(rsq) # saturation weighted by rsq
hsv[..., 2] = (rsq > rsq_threshold).astype(float) # value weighted by rsq

# convert hsv values of np array to rgb values (values assumed to be in range [0, 1])
rgb = colors.hsv_to_rgb(hsv)

# define alpha channel - which specifies the opacity for a color
# 0 = transparent = values with rsq below thresh and 1 = opaque = values above thresh
alpha_mask = (rsq <= rsq_threshold).T #why transpose? because of orientation of pycortex volume?
alpha = np.ones(alpha_mask.shape)
alpha[alpha_mask] = 0


# define Vertex images
images = {}

#contains RGBA colors for each voxel in a volumetric dataset
# vertex for polar angles
images['polar'] = cortex.VertexRGB(rgb[..., 0].T, 
                                 rgb[..., 1].T, 
                                 rgb[..., 2].T, 
                                 subject='fsaverage') #, alpha=alpha

# vertex for ecc
images['ecc'] = cortex.Vertex2D(eccentricity.T, alpha*10, 'fsaverage',
                           vmin=0, vmax=10,
                           vmin2=0, vmax2=1.0, cmap='BROYG_2D')
#images['ecc'] = cortex.Vertex2D(eccentricity.T, rsq.T, 'fsaverage',
#                           vmin=0, vmax=10,
#                           vmin2=rsq_threshold, vmax2=1.0, cmap='BROYG_2D')

# vertex for size
images['size'] = cortex.dataset.Vertex2D(size.T, alpha*10, 'fsaverage',
                           vmin=0, vmax=10,
                           vmin2=0, vmax2=1.0, cmap='BROYG_2D')
#images['size'] = cortex.Vertex2D(size.T, rsq.T, 'fsaverage',
#                           vmin=0, vmax=10,
#                           vmin2=rsq_threshold, vmax2=1.0, cmap='BROYG_2D')

# vertex for betas (amplitude?)
images['beta'] = cortex.Vertex2D(beta.T, rsq.T, 'fsaverage',
                           vmin=-2.5, vmax=2.5,
                           vmin2=rsq_threshold, vmax2=1.0, cmap='RdBu_r_alpha')

# vertex for baseline
images['baseline'] = cortex.Vertex2D(baseline.T, rsq.T, 'fsaverage',
                           vmin=-1, vmax=1,
                           vmin2=rsq_threshold, vmax2=1.0, cmap='RdBu_r_alpha')

# vertex for rsq
images['rsq'] = cortex.Vertex2D(rsq.T, alpha, 'fsaverage',
                           vmin=0, vmax=1.0,
                           vmin2=0, vmax2=1.0, cmap='Reds_cov')
#images['rsq'] = cortex.dataset.Vertex(rsq.T, 'fsaverage',
#                     vmin=0, vmax=np.max(rsq), cmap='Reds')


ds = cortex.Dataset(**images)
#cortex.webshow(ds, recache=True)
# Creates a static webGL MRI viewer in your filesystem
#web_path =os.path.join(analysis_params['cortex_dir'],'sub-{sj}'.format(sj=sj))
#cortex.webgl.make_static(outpath=web_path, data=ds) #, recache=True)#,template = 'cortex.html'){'polar':vrgba,'ecc':vecc}

# create flatmaps for different parameters and save png

flatmap_out = os.path.join(analysis_params['cortex_dir'],'sub-{sj}'.format(sj=sj),'flatmaps')
if not os.path.exists(flatmap_out): # check if path for outputs exist
        os.makedirs(flatmap_out)       # if not create it

# Save this flatmap
filename = os.path.join(flatmap_out,'flatmap_fsaverage_polar_angle.png')
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['polar'], recache=False,with_colorbar=False,with_curvature=True)

# Save this flatmap
filename = os.path.join(flatmap_out,'flatmap_fsaverage_eccentricity.png')
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['ecc'], recache=False,with_colorbar=True,with_curvature=True)

# Save this flatmap
filename = os.path.join(flatmap_out,'flatmap_fsaverage_size.png')
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['size'], recache=False,with_colorbar=True,with_curvature=True)

# Save this flatmap
filename = os.path.join(flatmap_out,'flatmap_fsaverage_rsquared.png')
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['rsq'], recache=False,with_colorbar=True,with_curvature=True)



