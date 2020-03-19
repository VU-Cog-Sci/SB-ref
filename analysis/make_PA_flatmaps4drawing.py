
# script to make polar angle flatmaps 
# with uniform color wheel and one-sided color wheel
# to better visualize borders and add to overlay (fsaverage_meridians)
# so I can actually draw them

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


# define participant number and open json parameter file
if len(sys.argv)<2: 
    raise NameError('Please add subject number (ex:01) '    
                    'as 1st argument in the command line!') 

else:   
    sj = str(sys.argv[1]).zfill(2) #fill subject number with 0 in case user forgets 

    with open('analysis_params.json','r') as json_file: 
            analysis_params = json.load(json_file)  


# used smoothed data (or not) for plots
with_smooth = 'True'#analysis_params['with_smooth']

# fit model to use (gauss or css)
fit_model = 'css' #analysis_params["fit_model"]
# if using estimates from iterative fit
iterative_fit = True #True

# total number of chunks that were fitted (per hemi)
total_chunks = analysis_params['total_chunks']

# define paths to save plots
figure_out = os.path.join(analysis_params['derivatives'],'figures','prf','final_fig',fit_model)

# set threshold for plotting
rsq_threshold = 0#0.17

if iterative_fit==True:
    figure_out = os.path.join(figure_out,'iterative','sub-{sj}'.format(sj=sj))
else:
    figure_out = os.path.join(figure_out,'grid','sub-{sj}'.format(sj=sj))
        
if not os.path.exists(figure_out): # check if path to save figures exists
    os.makedirs(figure_out) 

print('saving figures in %s'%figure_out)

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


# now mask them according to screen dimensions
masked_xx = []
masked_yy = []
masked_size = []
masked_beta = []
masked_baseline = []
masked_rsq = []
masked_ns = []
masked_polar_angle = []
masked_eccentricity = []

for w,subject in enumerate(estimates['subs']): # loop once if one subject, or for all subjects when sj 'all'

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
                                   estimates['baseline'][w],estimates['r2'][w],
                                   vert_lim_dva,hor_lim_dva,ns=estimates['ns'][w])

    masked_xx.append(new_estimates['x'])
    masked_yy.append(new_estimates['y'])
    masked_size.append(new_estimates['size'])
    masked_beta.append(new_estimates['beta'])
    masked_baseline.append(new_estimates['baseline'])
    masked_rsq.append(new_estimates['rsq'])
    masked_ns.append(new_estimates['ns'])
    
    # now construct polar angle and eccentricity values
    complex_location = new_estimates['x'] + new_estimates['y'] * 1j
    masked_polar_angle.append(np.angle(complex_location))
    masked_eccentricity.append(np.abs(complex_location))
    

## FLATMAPS ##
# compute median estimates (relevant for sub median)

masked_xx = np.nanmedian(masked_xx,axis=0) 
masked_yy = np.nanmedian(masked_yy,axis=0) 
masked_size = np.nanmedian(masked_size,axis=0)  
masked_beta = np.nanmedian(masked_beta,axis=0)  
masked_baseline = np.nanmedian(masked_baseline,axis=0)  
masked_rsq = np.nanmedian(masked_rsq,axis=0) 
masked_ns = np.nanmedian(masked_ns,axis=0) 


## remaining figures will be save in smoothed path (if its the case)

if with_smooth=='True':
    figure_out = os.path.join(figure_out,'smooth%d'%analysis_params['smooth_fwhm'])
    
    if not os.path.exists(figure_out): # check if path to save figures exists
        os.makedirs(figure_out) 

    print('saving figures in %s'%figure_out)
    
    # filename for smoothed estimates
    rsq_filename = os.path.join(figure_out,'rsq_pRF_fitmodel-%s_itertivefit-%s.npy'%(fit_model,str(iterative_fit)))
    size_filename = os.path.join(figure_out,'size_pRF_fitmodel-%s_itertivefit-%s.npy'%(fit_model,str(iterative_fit)))
    xx_filename = os.path.join(figure_out,'xx_pRF_fitmodel-%s_itertivefit-%s.npy'%(fit_model,str(iterative_fit)))
    yy_filename = os.path.join(figure_out,'yy_pRF_fitmodel-%s_itertivefit-%s.npy'%(fit_model,str(iterative_fit)))

    header_sub = '01' if sj=='median' else sj # to get header for smoothing
    
    # compute or load them (repetitive but good enough)
    
    ## RSQ
    if not os.path.isfile(rsq_filename):
        masked_rsq = smooth_nparray(os.path.join(analysis_params['post_fmriprep_outdir'], 'prf', 'sub-{sj}'.format(sj=str(header_sub).zfill(2))),
                           masked_rsq,
                           figure_out,
                           'rsq',
                           rsq_filename,
                           n_TR=83,
                           task='prf',
                           file_extension='_cropped_sg_psc.func.gii',
                           sub_mesh='fsaverage',
                           smooth_fwhm=analysis_params['smooth_fwhm'])
        np.save(rsq_filename,masked_rsq)
        
    elif os.path.isfile(rsq_filename):
        print('loading visual rsq from %s'%rsq_filename)
        masked_rsq = np.load(rsq_filename,allow_pickle=True)
        
    ## SIZE
    if not os.path.isfile(size_filename):
        masked_size = smooth_nparray(os.path.join(analysis_params['post_fmriprep_outdir'], 'prf', 'sub-{sj}'.format(sj=str(header_sub).zfill(2))),
                           masked_size,
                           figure_out,
                           'size',
                           size_filename,
                           n_TR=83,
                           task='prf',
                           file_extension='_cropped_sg_psc.func.gii',
                           sub_mesh='fsaverage',
                           smooth_fwhm=analysis_params['smooth_fwhm'])
        np.save(size_filename,masked_size)
        
    elif os.path.isfile(size_filename):
        print('loading visual size from %s'%size_filename)
        masked_size = np.load(size_filename,allow_pickle=True)
        
    ## X position
    if not os.path.isfile(xx_filename):
        masked_xx = smooth_nparray(os.path.join(analysis_params['post_fmriprep_outdir'], 'prf', 'sub-{sj}'.format(sj=str(header_sub).zfill(2))),
                           masked_xx,
                           figure_out,
                           'xx',
                           xx_filename,
                           n_TR=83,
                           task='prf',
                           file_extension='_cropped_sg_psc.func.gii',
                           sub_mesh='fsaverage',
                           smooth_fwhm=analysis_params['smooth_fwhm'])
        np.save(xx_filename,masked_xx)
        
    elif os.path.isfile(xx_filename):
        print('loading visual xx from %s'%xx_filename)
        masked_xx = np.load(xx_filename,allow_pickle=True)
        
    ## Y position
    if not os.path.isfile(yy_filename):
        masked_yy = smooth_nparray(os.path.join(analysis_params['post_fmriprep_outdir'], 'prf', 'sub-{sj}'.format(sj=str(header_sub).zfill(2))),
                           masked_yy,
                           figure_out,
                           'yy',
                           yy_filename,
                           n_TR=83,
                           task='prf',
                           file_extension='_cropped_sg_psc.func.gii',
                           sub_mesh='fsaverage',
                           smooth_fwhm=analysis_params['smooth_fwhm'])
        np.save(yy_filename,masked_yy)
        
    elif os.path.isfile(yy_filename):
        print('loading visual yy from %s'%yy_filename)
        masked_yy = np.load(yy_filename,allow_pickle=True)
        

complex_location = masked_xx + masked_yy * 1j
masked_polar_angle = np.angle(complex_location)
masked_eccentricity = np.abs(complex_location)    

images = {}

# normalize polar angles to have values in circle between 0 and 1
masked_polar_ang_norm = (masked_polar_angle + np.pi) / (np.pi * 2.0)

##### make PA flatmaps with non uniform color wheel ######
# shift radians in order to overrepresent red color
# useful to make NON-REPRESENTED retinotopic hemifield per hemisphere red
# then easier to define borders

# create HSV array, with PA values (-pi to pi) that were obtained from estimates
# saturation wieghted by a shifted distribution of RSQ (better visualization)
# value bolean (if I don't give it an rsq threshold then it's always 1)

hsv_angle = []
hsv_angle = np.ones((len(masked_rsq), 3))
hsv_angle[:, 0] = masked_polar_angle.copy()
hsv_angle[:, 1] = np.clip(masked_rsq / np.nanmax(masked_rsq) * 3, 0, 1)
#hsv_angle[:, 2] = masked_rsq > 0.14 #np.clip(masked_rsq / np.nanmax(masked_rsq) * 3, 0, 1)#masked_rsq > 0.12 #rsq_threshold

# get mid vertex index (diving hemispheres)
left_index = cortex.db.get_surfinfo('fsaverage').left.shape[0] 

### take angles from LH (thus RVF)##
angle_ = hsv_angle[:left_index, 0].copy()
angle_thresh = 3*np.pi/4 #value upon which to make it red for this hemifield (above it or below -angle will be red)

#normalized angles, between 0 and 1
hsv_angle[:left_index, 0] = np.clip((angle_ + angle_thresh)/(2*angle_thresh), 0, 1)

### take angles from RH (thus LVF) ##
angle_ = -hsv_angle[left_index:, 0].copy() # ATENÇÃO -> minus sign to flip angles vertically (then order of colors same for both hemispheres)

# sum 2pi to originally positive angles (now our trig circle goes from pi to 2pi to pi again, all positive)
angle_[hsv_angle[left_index:, 0] > 0] += 2 * np.pi

#normalized angles, between 0 and 1
angle_ = np.clip((angle_ + (angle_thresh-np.pi))/(2*angle_thresh), 0, 1) # ATENÇÃO -> we subtract -pi to angle thresh because now we want to rotate the whole thing -180 degrees

hsv_angle[left_index:, 0] = angle_.copy()
rgb_angle = []
rgb_angle = colors.hsv_to_rgb(hsv_angle)

# make alpha same as saturation, reduces clutter
alpha_angle = hsv_angle[:, 1]

images['angle_half_hemi'] = cortex.VertexRGB(rgb_angle[:, 0], rgb_angle[:, 1], rgb_angle[:, 2],
                                   alpha=alpha_angle,
                                   subject='fsaverage_meridians')
#cortex.quickshow(images['angle_half_hemi'],with_curvature=True,with_sulci=True,with_colorbar=False)

filename = os.path.join(figure_out,'flatmap_space-fsaverage_rsq-%0.2f_type-polar_angle_half_colorwheel.svg' %rsq_threshold)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['angle_half_hemi'], recache=True,with_colorbar=False,with_curvature=True,with_sulci=True)



# SAVE NORMAL PA (no rotation of hsv wheel, all colors equally represented)
hsv_angle = []
hsv_angle = np.ones((len(masked_rsq), 3))
hsv_angle[:, 0] = masked_polar_ang_norm.copy()
hsv_angle[:, 1] = np.clip(masked_rsq / np.nanmax(masked_rsq) * 2, 0, 1)
hsv_angle[:, 2] = masked_rsq > rsq_threshold
rgb_angle = colors.hsv_to_rgb(hsv_angle)
alpha_angle = hsv_angle[:, 1] #hsv_angle[:, 2]

images['angle'] = cortex.VertexRGB(rgb_angle[..., 0], rgb_angle[..., 1], rgb_angle[..., 2],
                                   alpha=alpha_angle,
                                   subject='fsaverage_meridians')
#cortex.quickshow(images['angle'],with_curvature=True,with_sulci=True,with_colorbar=False)

filename = os.path.join(figure_out,'flatmap_space-fsaverage_rsq-%0.2f_type-polar_angle_noshift.svg' %rsq_threshold)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['angle'], recache=True,with_colorbar=False,with_curvature=True,with_sulci=True)




# add to overlay
#cortex.utils.add_roi(images['angle_half_hemi'], name='polar_sub_%sj'%(sj), open_inkscape=False)

'''

## plot colorwheel and save in folder

resolution = 800
x, y = np.meshgrid(np.linspace(-1, 1, resolution), np.linspace(-1, 1, resolution))
radius = np.sqrt(x**2 + y**2)
polar_angle = np.arctan2(y, x)

polar_angle_circle = polar_angle.copy() # all polar angles calculated from our mesh
polar_angle_circle[radius > 1] = np.nan # then we're excluding all parts of bitmap outside of circle

# normal color wheel
cmap = plt.get_cmap('hsv')
norm = mpl.colors.Normalize(-np.pi, np.pi) # normalize between the point where we defined our color threshold

plt.imshow(polar_angle_circle, cmap=cmap, norm=norm,origin='lower')
plt.axis('off')
plt.savefig(os.path.join(analysis_params['derivatives'],'figures','prf','NEW_PA',fit_model,'color_wheel.svg'),dpi=100)

import matplotlib as mpl

cmap = plt.get_cmap('hsv')
norm = mpl.colors.Normalize(-angle_thresh, angle_thresh) # normalize between the point where we defined our color threshold

# for LH (RVF)
polar_angle_circle_left = polar_angle_circle.copy()
# between thresh angle make it red
polar_angle_circle_left[(polar_angle_circle_left < -angle_thresh) | (polar_angle_circle_left > angle_thresh)] = angle_thresh 
plt.imshow(polar_angle_circle_left, cmap=cmap, norm=norm,origin='lower') # origin lower because imshow flips it vertically, now in right order for VF
plt.axis('off')
plt.savefig(os.path.join(analysis_params['derivatives'],'figures','prf','NEW_PA',fit_model,'color_wheel_4LH-RVF.svg'),dpi=100)

# for RH (LVF)
polar_angle_circle_right = polar_angle_circle.copy()

polar_angle_circle_right = np.fliplr(polar_angle_circle_right)

polar_angle_circle_right[(polar_angle_circle_right < -.75 * np.pi) | (polar_angle_circle_right > 0.75 * np.pi)] = .75*np.pi
plt.imshow(polar_angle_circle_right, cmap=cmap, norm=norm,origin='lower')
plt.axis('off')
plt.savefig(os.path.join(analysis_params['derivatives'],'figures','prf','NEW_PA',fit_model,'color_wheel_4RH-LVF.svg'),dpi=100)

