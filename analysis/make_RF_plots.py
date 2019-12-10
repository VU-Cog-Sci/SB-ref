# make retino plots
# which will take as input subject number (01 or median)
# and produce relevant plots to check retinotopy
# save plots in figures/prf folder within derivatives


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
from prfpy.grid import Iso2DGaussianGridder
from prfpy.fit import Iso2DGaussianFitter

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


with_smooth = analysis_params['with_smooth']

# define paths
if with_smooth=='True':
    figure_out = os.path.join(analysis_params['derivatives'],'figures','prf','shift_crop','sub-{sj}'.format(sj=sj),'smooth%d'%analysis_params['smooth_fwhm'])
else:
    figure_out = os.path.join(analysis_params['derivatives'],'figures','prf','shift_crop','sub-{sj}'.format(sj=sj))


if not os.path.exists(figure_out): # check if path to save figures exists
    os.makedirs(figure_out) 


## Load PRF estimates ##
if sj=='median':
    estimates = median_iterative_pRFestimates(os.path.join(analysis_params['pRF_outdir'],'shift_crop'),with_smooth=False,exclude_subs=['sub-07']) # load unsmoothed estimates, will smooth later
    print('computed median estimates for %s excluded %s'%(str(estimates['subs']),'sub-07'))
    xx = estimates['x']
    yy = estimates['y']
    rsq = estimates['r2']
    size = estimates['size']
    beta = estimates['betas']
    baseline = estimates['baseline']
    
else:
    if with_smooth=='True':    
        median_path = os.path.join(analysis_params['pRF_outdir'],'shift_crop','sub-{sj}'.format(sj=sj),'run-median','smooth%d'%analysis_params['smooth_fwhm'],'iterative_fit')
    else:
        median_path = os.path.join(analysis_params['pRF_outdir'],'shift_crop','sub-{sj}'.format(sj=sj),'run-median','iterative_fit')

    estimates_list = [x for x in os.listdir(median_path) if x.endswith('iterative_output.npz')]
    estimates_list.sort() #sort to make sure pRFs not flipped

    estimates = []
    for _,val in enumerate(estimates_list) :
        print('appending %s'%val)
        estimates.append(np.load(os.path.join(median_path, val))) #save both hemisphere estimates in same array
        
    xx = np.concatenate((estimates[0]['it_output'][...,0],estimates[1]['it_output'][...,0]))
    yy = -np.concatenate((estimates[0]['it_output'][...,1],estimates[1]['it_output'][...,1])) # Need to do this for now, CHANGE ONCE BUG FIXED
    
    ###########################

    size = np.concatenate((estimates[0]['it_output'][...,2],estimates[1]['it_output'][...,2]))
    beta = np.concatenate((estimates[0]['it_output'][...,3],estimates[1]['it_output'][...,3]))
    baseline = np.concatenate((estimates[0]['it_output'][...,4],estimates[1]['it_output'][...,4]))

    rsq = np.concatenate((estimates[0]['it_output'][...,5],estimates[1]['it_output'][...,5])) 



# set limits for xx and yy, forcing it to be within the screen boundaries

vert_lim_dva = (analysis_params['screenRes'][-1]/2) * dva_per_pix(analysis_params['screen_width'],analysis_params['screen_distance'],analysis_params['screenRes'][-1])
hor_lim_dva = (analysis_params['screenRes'][0]/2) * dva_per_pix(analysis_params['screen_width'],analysis_params['screen_distance'],analysis_params['screenRes'][-1])

# make new variables that are masked (cebter of RF within screen limits and only positive pRFs)
print('masking variables to be within screen and only show positive RF')
new_estimates = mask_estimates(xx,yy,size,beta,baseline,rsq,vert_lim_dva,hor_lim_dva)

masked_xx = new_estimates['x']
masked_yy = new_estimates['y']
masked_size = new_estimates['size']
masked_beta = new_estimates['beta']
masked_baseline = new_estimates['baseline']
masked_rsq = new_estimates['rsq']


# to make smoothed plots for median subject, need to convert estimates into gii
# and then smooth images
if sj=='median' and with_smooth=='True':
    # empty array to save smoothed filenames
    smooth_filename = []

    # load random subject, just to get header to save median estimates as gii

    filepath = glob.glob(os.path.join(analysis_params['post_fmriprep_outdir'], 'prf', 'sub-11', '*'))
    print('loading first run to get header info from %s' % os.path.split(filepath[0])[0])

    # last part of filename to use
    file_extension = 'cropped_sg_psc.func.gii'

    # load first run of subject
    filename = [run for run in filepath if 'prf' in run and 'fsaverage' in run and 'run-01' in run and run.endswith(file_extension)]
    filename.sort()

    # path to save fits, for testing
    out_dir = figure_out

    for field in ['hemi-L', 'hemi-R']:
        hemi = [h for h in filename if field in h]
        
        num_vert_hemi = int(masked_xx.shape[0]/2) #number of vertices in one hemisphere
        median_filename = 'sub-{sj}'.format(sj=sj)+'_task-prf_run-median_space-fsaverage_'+field+'_'+file_extension
        
        if field=='hemi-L':
            xx_4smoothing = masked_xx[0:num_vert_hemi]
            yy_4smoothing = masked_yy[0:num_vert_hemi]
            size_4smoothing = masked_size[0:num_vert_hemi]
            beta_4smoothing = masked_beta[0:num_vert_hemi]
            baseline_4smoothing = masked_baseline[0:num_vert_hemi]
            rsq_4smoothing = masked_rsq[0:num_vert_hemi]
            
        else:
            xx_4smoothing = masked_xx[num_vert_hemi::]
            yy_4smoothing = masked_yy[num_vert_hemi::]
            size_4smoothing = masked_size[num_vert_hemi::]
            beta_4smoothing = masked_beta[num_vert_hemi::]
            baseline_4smoothing = masked_baseline[num_vert_hemi::]
            rsq_4smoothing = masked_rsq[num_vert_hemi::]
          
        # reunite them in same array
        estimates4smoothing = {'xx':xx_4smoothing,'yy':yy_4smoothing,'size':size_4smoothing,
                               'beta':beta_4smoothing,'baseline':baseline_4smoothing,'rsq':rsq_4smoothing}
        
        img_load = nb.load(hemi[0]) # load run just to get header
        
        for _,arr in enumerate(estimates4smoothing):
            new_filename = os.path.join(out_dir,median_filename.replace('.func.gii','_estimates-%s.func.gii'%arr))
            
            print('saving %s'%new_filename)
            est_array_tiled = np.tile(estimates4smoothing[arr][np.newaxis,...],(83,1)) # NEED TO DO THIS 4 MGZ to actually be read (header is of func file)
            darrays = [nb.gifti.gifti.GiftiDataArray(d) for d in est_array_tiled]
            estimates_gii = nb.gifti.gifti.GiftiImage(header=img_load.header,
                                               extra=img_load.extra,
                                               darrays=darrays) # need to save as gii
            
            nb.save(estimates_gii,new_filename)
            
            _,smo_estimates_path = smooth_gii(new_filename,out_dir,fwhm=analysis_params['smooth_fwhm'])
            
            smooth_filename.append(smo_estimates_path)
            print('saving %s'%smo_estimates_path)
        
    # load files save as new masked estimates, to be analogous to other situations
    smooth_filename.sort()
    smooth_xx = []
    smooth_yy = []
    smooth_size = []
    smooth_beta = []
    smooth_baseline = []
    smooth_rsq = []

    for _,name in enumerate(smooth_filename): # not elegant but works
        img_load = nb.load(name)
        if '_estimates-xx_smooth%d.func.gii'%analysis_params['smooth_fwhm'] in name:
            smooth_xx.append(np.array([img_load.darrays[i].data for i in range(len(img_load.darrays))]))
        elif '_estimates-yy_smooth%d.func.gii'%analysis_params['smooth_fwhm'] in name:
            smooth_yy.append(np.array([img_load.darrays[i].data for i in range(len(img_load.darrays))]))
        elif '_estimates-size_smooth%d.func.gii'%analysis_params['smooth_fwhm'] in name:
            smooth_size.append(np.array([img_load.darrays[i].data for i in range(len(img_load.darrays))]))
        elif '_estimates-beta_smooth%d.func.gii'%analysis_params['smooth_fwhm'] in name:
            smooth_beta.append(np.array([img_load.darrays[i].data for i in range(len(img_load.darrays))]))
        elif '_estimates-baseline_smooth%d.func.gii'%analysis_params['smooth_fwhm'] in name:
            smooth_baseline.append(np.array([img_load.darrays[i].data for i in range(len(img_load.darrays))]))
        elif '_estimates-rsq_smooth%d.func.gii'%analysis_params['smooth_fwhm'] in name:
            smooth_rsq.append(np.array([img_load.darrays[i].data for i in range(len(img_load.darrays))]))

    # concatenate both hemis in one array
    masked_xx = np.concatenate((smooth_xx[0][0],smooth_xx[1][0]))  
    masked_yy = np.concatenate((smooth_yy[0][0],smooth_yy[1][0])) 
    masked_size = np.concatenate((smooth_size[0][0],smooth_size[1][0])) 
    masked_beta = np.concatenate((smooth_beta[0][0],smooth_beta[1][0])) 
    masked_baseline = np.concatenate((smooth_baseline[0][0],smooth_baseline[1][0])) 
    masked_rsq = np.concatenate((smooth_rsq[0][0],smooth_rsq[1][0])) 
    

# save masked estimates in numpy array to load later
# saved in same folder as figures (for now, might change that later)

masked_estimates_filename = os.path.join(figure_out,'masked_estimates.npz')
np.savez(masked_estimates_filename,
              masked_xx = masked_xx,
              masked_yy = masked_yy,
              masked_size = masked_size,
              masked_beta = masked_beta,
              masked_baseline = masked_baseline,
              masked_rsq = masked_rsq
              )

# now construct polar angle and eccentricity values
rsq_threshold = analysis_params['rsq_threshold']

complex_location = masked_xx + masked_yy * 1j
masked_polar_angle = np.angle(complex_location)
masked_eccentricity = np.abs(complex_location)

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
alpha_mask = np.array([True if val<= rsq_threshold or np.isnan(val) else False for _,val in enumerate(masked_rsq)]).T #why transpose? because of orientation of pycortex volume?

# create alpha array weighted by rsq values
alpha = np.sqrt(masked_rsq.copy())#np.ones(alpha_mask.shape)
alpha[alpha_mask] = np.nan

# create alpha array with nan = transparent = values with rsq below thresh and 1 = opaque = values above thresh
alpha_ones = np.ones(alpha_mask.shape)
alpha_ones[alpha_mask] = np.nan


images = {}

images['polar'] = cortex.VertexRGB(rgb[..., 0].T, 
                                 rgb[..., 1].T, 
                                 rgb[..., 2].T, 
                                 subject='fsaverage_gross', alpha=alpha)
#cortex.quickshow(images['polar'],with_curvature=True,with_sulci=True,with_colorbar=False)
filename = os.path.join(figure_out,'flatmap_space-fsaverage_rsq-%0.2f_type-polar_angle.svg' %rsq_threshold)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['polar'], recache=True,with_colorbar=False,with_curvature=True,with_sulci=True)

images['polar_noalpha'] = cortex.VertexRGB(rgb[..., 0].T, 
                                 rgb[..., 1].T, 
                                 rgb[..., 2].T, 
                                 subject='fsaverage_gross', alpha=alpha_ones)
#cortex.quickshow(images['polar_noalpha'],with_curvature=True,with_sulci=True,with_colorbar=False)
filename = os.path.join(figure_out,'flatmap_space-fsaverage_rsq-%0.2f_type-polar_angle_noalpha.svg' %rsq_threshold)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['polar_noalpha'], recache=True,with_colorbar=False,with_curvature=True,with_sulci=True)


# vertex for rsq
images['rsq'] = cortex.Vertex2D(masked_rsq.T, alpha_ones, 'fsaverage_gross',
                           vmin=0, vmax=1.0,
                           vmin2=0, vmax2=1.0, cmap='Reds_cov')
#cortex.quickshow(images['rsq'],with_curvature=True,with_sulci=True)
filename = os.path.join(figure_out,'flatmap_space-fsaverage_rsq-%0.2f_type-rsquared.svg' %rsq_threshold)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['rsq'], recache=True,with_colorbar=True,with_curvature=True,with_sulci=True)


# vertex for ecc
ecc4plot = masked_eccentricity.copy()
ecc4plot[alpha_mask] = np.nan

images['ecc'] = cortex.Vertex(ecc4plot.T, 'fsaverage_gross',
                           vmin=0, vmax=analysis_params['max_eccen'],
                           cmap='J4')
#cortex.quickshow(images['ecc'],with_curvature=True,with_sulci=True)
# Save this flatmap
filename = os.path.join(figure_out,'flatmap_space-fsaverage_rsq-%0.2f_type-eccentricity.svg' %rsq_threshold)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['ecc'], recache=True,with_colorbar=True,with_curvature=True,with_sulci=True)


# vertex for ecc
size4plot = masked_size.copy()
size4plot[alpha_mask] = np.nan

# vertex for size
images['size'] = cortex.dataset.Vertex(size4plot.T, 'fsaverage_gross',
                           vmin=0, vmax=analysis_params['max_size'],
                           cmap='J4')
#cortex.quickshow(images['size'],with_curvature=True,with_sulci=True)
filename = os.path.join(figure_out,'flatmap_space-fsaverage_rsq-%0.2f_type-size.svg' %rsq_threshold)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['size'], recache=True,with_colorbar=True,with_curvature=True,with_sulci=True)


# get vertices for subject fsaverage
# for later plotting

ROIs = [['V1','V2','V3'],'V1','V2','V3','sPCS','iPCS'] # list will combine those areas, string only accounts for 1 ROI

roi_verts = {} #empty dictionary 
for i,val in enumerate(ROIs):
    
    if type(val)==str: # if string, we can directly get the ROI vertices  
        roi_verts[val] = cortex.get_roi_verts('fsaverage_gross',val)[val]

    else: # if list
        indice = []
        for w in range(len(ROIs[0])): # load vertices for each region of list
            indice.append(cortex.get_roi_verts('fsaverage_gross',val[w])[val[w]])
        
        roi_verts[str(val)] = np.hstack(indice)


# plot
for idx,roi in enumerate(ROIs):
    
    f, s = plt.subplots(1, 2, figsize=(24,48))

    if type(roi)!=str: # if list
        roi = str(roi)
    # get datapoints for RF only belonging to roi
    new_rsq = masked_rsq[roi_verts[roi]]

    new_xx = masked_xx[roi_verts[roi]]
    new_yy = masked_yy[roi_verts[roi]]
    new_size = masked_size[roi_verts[roi]]
    
    new_ecc = masked_eccentricity[roi_verts[roi]]
    new_polar_angle = masked_polar_angle[roi_verts[roi]]
    
    #####
    # normalize polar angles to have values in circle between 0 and 1
    new_pa_norm = (new_polar_angle + np.pi) / (np.pi * 2.0)

    # use "resto da divisão" so that 1 == 0 (because they overlapp in circle)
    # why have an offset?
    new_pa_norm = np.fmod(new_pa_norm+angle_offset, 1.0)

    # convert angles to colors, using correlations as weights
    hsv_colors = np.zeros(list(new_pa_norm.shape) + [3])
    hsv_colors[..., 0] = new_pa_norm # different hue value for each angle
    hsv_colors[..., 1] = 1#(new_rsq > rsq_threshold).astype(float)#  np.ones_like(rsq) # saturation weighted by rsq
    hsv_colors[..., 2] = 1#(new_rsq > rsq_threshold).astype(float) # value weighted by rsq

    # convert hsv values of np array to rgb values (values assumed to be in range [0, 1])
    rgb_col = colors.hsv_to_rgb(hsv_colors)
    rgba_colors = np.zeros((rgb_col.shape[0], 4))
    rgba_colors[:,0] = rgb_col[:,0]
    rgba_colors[:,1] = rgb_col[:,1]
    rgba_colors[:,2] = rgb_col[:,2]
    rgba_colors[:,3] = normalize(new_rsq)/2 # alpha will be normalized rsq scaled in half
    ######## 
    
    s[0].set_title('%s pRFs in visual field'%roi)
    s[0].set_xlim([-analysis_params["max_eccen"],analysis_params["max_eccen"]])
    s[0].set_ylim([-analysis_params["max_eccen"],analysis_params["max_eccen"]])
    s[0].axvline(0, -15, 15, c='k', lw=0.25)
    s[0].axhline(0, -15, 15, c='k', lw=0.25)
    # new way to plot - like this I'm sure of positions of RF and radius of circle scaled as correct size
    # new way to plot - like this I'm sure of positions of RF and radius of circle scaled as correct size
    plot_ind = [k for k in range(len(new_xx))]
    random.shuffle(plot_ind) # randomize indices to plot, avoids biases
    for _,w in enumerate(plot_ind):
        if new_rsq[w]>rsq_threshold:
            s[0].add_artist(plt.Circle((new_xx[w], new_yy[w]), radius=new_size[w], color=rgba_colors[w], fill=True))
        #s[0].scatter(new_xx[w], new_yy[w], s=new_size[w])

    s[0].set_xlabel('horizontal space [dva]')
    s[0].set_ylabel('vertical space [dva]')
    
    # Create a Rectangle patch
    rect = patches.Rectangle((-hor_lim_dva,-vert_lim_dva),hor_lim_dva*2,vert_lim_dva*2,linewidth=1,linestyle='--',edgecolor='k',facecolor='none',zorder=10)

    # Add the patch to the Axes
    s[0].add_patch(rect)

    s[1].set_title('%s pRF size vs eccentricity'%roi)
    s[1].set_xlim([0,15])
    s[1].set_ylim([0,analysis_params["max_size"]])
    s[1].set_xlabel('pRF eccentricity [dva]')
    s[1].set_ylabel('pRF size [dva]')
    s[1].scatter(new_ecc[new_rsq>rsq_threshold], new_size[new_rsq>rsq_threshold]) #color=rgba_colors, edgecolors=edgecolors, linewidths=2);  # this size is made up - beware.
    
    # make sure that plots have proportional size
    s[0].set(adjustable='box-forced', aspect='equal')
    s[1].set(adjustable='box-forced', aspect='equal')
    f.savefig(os.path.join(figure_out,'RF_scatter_ROI-%s_rsq-%0.2f.svg'%(roi,rsq_threshold)), dpi=100,bbox_inches = 'tight')

    # plot left and right hemi coverage separetly
    
    if roi!=str(['V1','V2','V3']): # if single roi
        f, s = plt.subplots(1, 2, figsize=(24,48))
        
        # get roi indices for each hemisphere
        left_roi_verts = roi_verts[roi][roi_verts[roi]<int(len(masked_rsq)/2)]
        right_roi_verts = roi_verts[roi][roi_verts[roi]>=int(len(masked_rsq)/2)]

        # LEFT VISUAL FIELD
        s[0].set_title('%s pRFs in visual field for left hemisphere'%roi)
        s[0].set_xlim([-analysis_params["max_eccen"],analysis_params["max_eccen"]])
        s[0].set_ylim([-analysis_params["max_eccen"],analysis_params["max_eccen"]])
        s[0].axvline(0, -15, 15, c='k', lw=0.25)
        s[0].axhline(0, -15, 15, c='k', lw=0.25)

        # RIGHT VISUAL FIELD
        s[1].set_title('%s pRFs in visual field for right hemisphere'%roi)
        s[1].set_xlim([-analysis_params["max_eccen"],analysis_params["max_eccen"]])
        s[1].set_ylim([-analysis_params["max_eccen"],analysis_params["max_eccen"]])
        s[1].axvline(0, -15, 15, c='k', lw=0.25)
        s[1].axhline(0, -15, 15, c='k', lw=0.25)

        # new way to plot - like this I'm sure of positions of RF and radius of circle scaled as correct size
        # new way to plot - like this I'm sure of positions of RF and radius of circle scaled as correct size
        plot_ind = [k for k in range(len(roi_verts[roi]))]
        random.shuffle(plot_ind) # randomize indices to plot, avoids biases
        for _,w in enumerate(plot_ind):
            if new_rsq[w]>rsq_threshold:
                if w<len(left_roi_verts): # then its left hemisphere
                    s[0].add_artist(plt.Circle((new_xx[w], new_yy[w]), radius=new_size[w], color=rgba_colors[w], fill=True))
                else:
                    s[1].add_artist(plt.Circle((new_xx[w], new_yy[w]), radius=new_size[w], color=rgba_colors[w], fill=True))

        s[0].set_xlabel('horizontal space [dva]')
        s[0].set_ylabel('vertical space [dva]')

        s[1].set_xlabel('horizontal space [dva]')
        s[1].set_ylabel('vertical space [dva]')

        # Create a Rectangle patch
        rect = patches.Rectangle((-hor_lim_dva,-vert_lim_dva),hor_lim_dva*2,vert_lim_dva*2,linewidth=1,linestyle='--',edgecolor='k',facecolor='none',zorder=10)
        rect2 = patches.Rectangle((-hor_lim_dva,-vert_lim_dva),hor_lim_dva*2,vert_lim_dva*2,linewidth=1,linestyle='--',edgecolor='k',facecolor='none',zorder=10)

        # Add the patch to the Axes
        s[0].add_patch(rect)
        # Add the patch to the Axes
        s[1].add_patch(rect2)

        # make sure that plots have proportional size
        s[0].set(adjustable='box-forced', aspect='equal')
        s[1].set(adjustable='box-forced', aspect='equal')

        f.savefig(os.path.join(figure_out,'RF_scatter_LRhemi_ROI-%s_rsq-%0.2f.svg'%(roi,rsq_threshold)), dpi=100,bbox_inches = 'tight')

    
#plt.show() 


# now do single voxel fits, choosing voxels with highest rsq 
# from each ROI (early visual vs sPCS vs iPCS)
# and plot all in same figure (usefull for fig1)

if sj != 'median': # doesn't work for median subject
    # path to functional files
    filepath = glob.glob(os.path.join(analysis_params['post_fmriprep_outdir'], 'prf', 'sub-{sj}'.format(sj=sj), '*'))
    print('functional files from %s' % os.path.split(filepath[0])[0])

    # last part of filename to use
    file_extension = 'cropped_sg_psc.func.gii'

    # list of functional files (5 runs)
    filename = [run for run in filepath if 'prf' in run and 'fsaverage' in run and run.endswith(file_extension)]
    filename.sort()

    # path to save fits, for testing
    out_dir = os.path.join(analysis_params['pRF_outdir'],'shift_crop','sub-{sj}'.format(sj=sj),'tests')

    if not os.path.exists(out_dir):  # check if path exists
        os.makedirs(out_dir)


    # loads median run functional files and saves the absolute path name in list
    med_gii = [] 
    for field in ['hemi-L', 'hemi-R']:
        hemi = [h for h in filename if field in h]

        # set name for median run (now numpy array)
        med_file = os.path.join(out_dir, re.sub(
            'run-\d{2}_', 'run-median_', os.path.split(hemi[0])[-1]))
        # if file doesn't exist
        if not os.path.exists(med_file):
            med_gii.append(median_gii(hemi, out_dir))  # create it
            print('computed %s' % (med_gii))
        else:
            med_gii.append(med_file)
            print('median file %s already exists, skipping' % (med_gii))

    # load data for median run, one hemisphere 
    hemi = ['hemi-L','hemi-R']

    data = []
    for _,h in enumerate(hemi):
        gii_file = med_gii[0] if h == 'hemi-L' else  med_gii[1]
        print('using %s' %gii_file)
        data.append(np.array(surface.load_surf_data(gii_file)))

    data = np.vstack(data) # will be (vertex, TR)
    
    
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
    fit_model = analysis_params["fit_model"]

    TR = analysis_params["TR"]

    hrf = utilities.spm_hrf(0,TR)

    # make stimulus object, which takes an input design matrix and sets up its real-world dimensions
    prf_stim = PRFStimulus2D(screen_size_cm=analysis_params["screen_width"], 
                             screen_distance_cm=analysis_params["screen_distance"], 
                             design_matrix=prf_dm, 
                             TR=TR)

    # sets up stimulus and hrf for this gridder
    gg = Iso2DGaussianGridder(stimulus=prf_stim,
                              hrf=hrf,
                              filter_predictions=False,
                              window_length=analysis_params["sg_filt_window_length"],
                              polyorder=analysis_params["sg_filt_polyorder"],
                              highpass=False)



    ##### #################################### #############
    
    ############ SHOW FIT FOR SINGLE VOXEL AND PLOT IT OVER DATA ##############

    # times where bar is on screen [1st on, last on, 1st on, last on, etc] 
    bar_onset = (np.array([14,22,25,41,55,71,74,82])-analysis_params['crop_pRF_TR'])*TR

    # get single voxel data from ROI where rsq is max 
    # get datapoints for RF only belonging to roi
    for idx,roi in enumerate(ROIs):
        if type(roi)!=str: # if list
            print('skipping list ROI %s for timeseries plot'%str(roi))
        else:
            new_rsq = masked_rsq[roi_verts[roi]]

            new_xx = masked_xx[roi_verts[roi]]
            new_yy = masked_yy[roi_verts[roi]]
            new_size = masked_size[roi_verts[roi]]

            new_beta = masked_beta[roi_verts[roi]]
            new_baseline = masked_baseline[roi_verts[roi]]

            new_ecc = masked_eccentricity[roi_verts[roi]]
            new_polar_angle = masked_polar_angle[roi_verts[roi]]


            new_data = data[roi_verts[roi]] # data from ROI

            new_index =np.where(new_rsq==np.nanmax(new_rsq))[0][0]# index for max rsq within ROI

            timeseries = new_data[new_index]

            model_it_prfpy = gg.return_single_prediction(new_xx[new_index],-new_yy[new_index],new_size[new_index], #because we also added - before, so will be actual yy val from fit
                                                         beta=new_beta[new_index],baseline=new_baseline[new_index])
            model_it_prfpy = model_it_prfpy[0]

            print('voxel %d of ROI %s , rsq of fit is %.3f' %(new_index,roi,new_rsq[new_index]))

            # plot data with model
            time_sec = np.linspace(0,len(timeseries)*TR,num=len(timeseries)) # array with 90 timepoints, in seconds
            fig= plt.figure(figsize=(15,7.5),dpi=100)
            plt.plot(time_sec,model_it_prfpy,c='#db3050',lw=3,label='prf model',zorder=1)
            plt.scatter(time_sec,timeseries, marker='v',c='k',label='data')
            plt.xlabel('Time (s)',fontsize=18)
            plt.ylabel('BOLD signal change (%)',fontsize=18)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.xlim(0,len(timeseries)*TR)
            plt.title('voxel %d of ROI %s , rsq of fit is %.3f' %(new_index,roi,new_rsq[new_index]))

            # plot axis vertical bar on background to indicate stimulus display time
            ax_count = 0
            for h in range(4):
                plt.axvspan(bar_onset[ax_count], bar_onset[ax_count+1]+TR, facecolor='r', alpha=0.1)
                ax_count += 2
            
            plt.legend(loc=0)

            fig.savefig(os.path.join(figure_out,'pRF_singvoxfit_timeseries_%s_rsq-%0.2f.svg'%(roi,rsq_threshold)), dpi=100,bbox_inches = 'tight')


            ################# get receptive field for that voxel ###################################

            # add this so then I can see which bar passes correspond to model peaks
            # hence check if it makes sense
            # plot RF for voxel and bar passes corresponding to model peaks
            sig_peaks = scipy.signal.find_peaks(model_it_prfpy,height=0.5) #find peaks
            print('peaks for roi %s'%roi)
            print(sig_peaks)

            fig = plt.figure(figsize=(24,48),constrained_layout=True)
            outer = gridspec.GridSpec(1, 2, wspace=0.4)

            for i in range(2):
                if i == 0: #first plot, one subplot
                    inner = gridspec.GridSpecFromSubplotSpec(1,1,
                                    subplot_spec=outer[i])
                    ax = plt.Subplot(fig, inner[0])
                    ax.set_title('RF position for voxel %d of %s'%(new_index,roi))
                    ax.set_xlim([-analysis_params["max_eccen"],analysis_params["max_eccen"]])
                    ax.set_ylim([-analysis_params["max_eccen"],analysis_params["max_eccen"]])
                    ax.axvline(0, -15, 15, c='k', lw=0.25)
                    ax.axhline(0, -15, 15, c='k', lw=0.25)
                    ax.add_artist(plt.Circle((new_xx[new_index],new_yy[new_index]), new_size[new_index], color='r',alpha=new_rsq[new_index]))
                    ax.set(adjustable='box-forced', aspect='equal')   

                    fig.add_subplot(ax)

                else: #second plot with 4 subplots
                    inner = gridspec.GridSpecFromSubplotSpec(1,2,#2, 2,
                                    subplot_spec=outer[i])

                    # plot bar pass for peaks
                    k = 0
                    for j in range(2):
                        inner1 = gridspec.GridSpecFromSubplotSpec(2,1, subplot_spec=inner[j]) 

                        for w in range(2):
                            ax = plt.Subplot(fig, inner1[w])#inner[j,w])
                            ax.imshow(prf_dm[:,:,sig_peaks[0][k]-3].T) # subtract 3 TRs because hemodynamic response takes 4-6s to peak = 3TRs * 1.6 = 4.8s (or so)
                            ax.set_title('bar pass TR = %d'%(sig_peaks[0][k]-3))
                            ax.set(adjustable='box-forced',aspect='equal') 
                            fig.add_subplot(ax)
                            k += 1
          
            fig.savefig(os.path.join(figure_out,'RF_singvoxfit_%s_rsq-%0.2f.svg'%(roi,rsq_threshold)), dpi=100,bbox_inches = 'tight')

            #############################


    # redo vertice list (repetitive but quick fix for now)
    ROIs = [['V1','V2','V3'],'sPCS','iPCS']
    
    # Plot median timeseries, just to check for bias
    fig= plt.figure(figsize=(15,7.5))

    for i in range(len(ROIs)):
        c = ['r','g','b']
        new_data = data[roi_verts[str(ROIs[i])]] # data from ROI

        plt.plot(time_sec,np.nanmedian(new_data[rsq[roi_verts[str(ROIs[i])]]>rsq_threshold],axis=0),c=c[i],label=str(ROIs[i]))
        plt.legend(loc=0)
    plt.xlabel('Time (s)',fontsize=18)
    plt.ylabel('BOLD signal change (%)',fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(0,90*TR)
    # plot axis vertical bar on background to indicate stimulus display time
    ax_count = 0
    for h in range(4):
        plt.axvspan(bar_onset[ax_count], bar_onset[ax_count+1]+TR, facecolor='r', alpha=0.1)
        ax_count += 2

    #plt.show()    
    fig.savefig(os.path.join(figure_out,'pRF_median_timeseries_allROIs_rsq-%0.2f.svg'%rsq_threshold), dpi=100,bbox_inches = 'tight')
    
    
    ROIs = ['V1','sPCS']
    roi_verts = {} #empty dictionary 
    for i,val in enumerate(ROIs):   
        if type(val)==str: # if string, we can directly get the ROI vertices  
            roi_verts[val] = cortex.get_roi_verts('fsaverage_gross',val)[val]

    red_color = ['#591420','#d12e4c']
    data_color = ['#262626','#8a8a8a']

    fig, axis = plt.subplots(1,figsize=(15,7.5),dpi=100)
    for idx,roi in enumerate(ROIs):
        if type(roi)!=str: # if list
            print('skipping list ROI %s for timeseries plot'%str(roi))
        else:
            new_rsq = masked_rsq[roi_verts[roi]]

            new_xx = masked_xx[roi_verts[roi]]
            new_yy = masked_yy[roi_verts[roi]]
            new_size = masked_size[roi_verts[roi]]

            new_beta = masked_beta[roi_verts[roi]]
            new_baseline = masked_baseline[roi_verts[roi]]

            new_ecc = masked_eccentricity[roi_verts[roi]]
            new_polar_angle = masked_polar_angle[roi_verts[roi]]


            new_data = data[roi_verts[roi]] # data from ROI

            new_index =np.where(new_rsq==np.nanmax(new_rsq))[0][0]# index for max rsq within ROI

            timeseries = new_data[new_index]

            model_it_prfpy = gg.return_single_prediction(new_xx[new_index],-new_yy[new_index],new_size[new_index], #because we also added - before, so will be actual yy val from fit
                                                         beta=new_beta[new_index],baseline=new_baseline[new_index])
            model_it_prfpy = model_it_prfpy[0]

            print('voxel %d of ROI %s , rsq of fit is %.3f' %(new_index,roi,new_rsq[new_index]))

            # plot data with model
            time_sec = np.linspace(0,len(timeseries)*TR,num=len(timeseries)) # array with 90 timepoints, in seconds
            
            # instantiate a second axes that shares the same x-axis
            if roi == 'sPCS': axis = axis.twinx() 
            
            # plot data with model
            axis.plot(time_sec,model_it_prfpy,c=red_color[idx],lw=3,label=roi+', R$^2$=%.2f'%new_rsq[new_index],zorder=1)
            axis.scatter(time_sec,timeseries, marker='v',s=15,c=red_color[idx])#,label=roi)
            axis.set_xlabel('Time (s)',fontsize=18)
            axis.set_ylabel('BOLD signal change (%)',fontsize=18)
            axis.tick_params(axis='both', labelsize=14)
            axis.tick_params(axis='y', labelcolor=red_color[idx])
            axis.set_xlim(0,len(timeseries)*TR)
            plt.gca().set_ylim(bottom=0)

            #plt.title('voxel %d (%s) , MSE = %.3f, rsq = %.3f' %(vertex[i],task[i],mse,r2))
            
            # to align axis centering it at 0
            if idx == 0:
                if sj=='11':
                    axis.set_ylim(-3,9)
                ax1 = axis
            else:
                if sj=='11':
                    axis.set_ylim(-1.5,3.5)
                align_yaxis(ax1, 0, axis, 0)

            #axis.axhline(y=0, xmin=0, xmax=len(timeseries)*TR,linestyle='--',c=red_color[idx])

            #if roi == 'sPCS':
            #    axis.set_ylim(-2,5) 
            #else:
            #    axis.set_ylim(-4,10)
            #plt.title('voxel %d (%s) , MSE = %.3f, rsq = %.3f' %(vertex[i],task[i],mse,r2))
            
            if idx == 0:
                handles,labels = axis.axes.get_legend_handles_labels()
            else:
                a,b = axis.axes.get_legend_handles_labels()
                handles = handles+a
                labels = labels+b

            # plot axis vertical bar on background to indicate stimulus display time
            ax_count = 0
            for h in range(4):
                plt.axvspan(bar_onset[ax_count], bar_onset[ax_count+1]+TR, facecolor=red_color[idx], alpha=0.1)
                ax_count += 2

            
    axis.legend(handles,labels,loc='upper left')  # doing this to guarantee that legend is how I want it   

    fig.savefig(os.path.join(figure_out,'pRF_singvoxfit_timeseries_%s_rsq-%0.2f.svg'%(str(ROIs),rsq_threshold)), dpi=100,bbox_inches = 'tight')


# make plot to show relationship between ecc and size for different regions
# redo vertice list (repetitive but quick fix for now)
ROIs = ['V1','V2','V3','sPCS','iPCS']#[['V1','V2','V3'],'sPCS','iPCS']
roi_verts = {} #empty dictionary 
for i,val in enumerate(ROIs):   
    if type(val)==str: # if string, we can directly get the ROI vertices  
        roi_verts[val] = cortex.get_roi_verts('fsaverage_gross',val)[val]
        
# create empty dataframe to store all relevant values for rois
all_roi = pd.DataFrame(columns=['mean_ecc','mean_ecc_std','mean_size','mean_size_std','roi'])
n_bins = 10
min_ecc = 0.25
max_ecc = 4

if sj == 'median':
    all_estimates = append_pRFestimates(os.path.join(analysis_params['pRF_outdir'],'shift_crop'),with_smooth=False,exclude_subs=['sub-07'])

for idx,roi in enumerate(ROIs):
    
    df = pd.DataFrame(columns=['ecc','size','rsq'])

    if sj == 'median':
        for s in range(np.array(all_estimates['x']).shape[0]): # loop over all subjects that were appended
        
            #print('masking variables to be within screen and only show positive RF')
            sub_masked_estimates = mask_estimates(all_estimates['x'][s],all_estimates['y'][s],
                                                  all_estimates['size'][s],all_estimates['betas'][s],
                                                  all_estimates['baseline'][s],all_estimates['r2'][s],vert_lim_dva,hor_lim_dva)
        
            # get datapoints for RF only belonging to roi
            new_size = sub_masked_estimates['size'][roi_verts[str(roi)]]
            new_ecc = np.abs(sub_masked_estimates['x'][roi_verts[str(roi)]] + sub_masked_estimates['y'][roi_verts[str(roi)]] * 1j)        
            new_rsq = sub_masked_estimates['rsq'][roi_verts[str(roi)]]
            
            # define indices of voxels within region to plot
            # with rsq > 0.15, and where value not nan, ecc values between 0.25 and 4
            indices4plot = np.where((new_ecc >= min_ecc) & (new_ecc<= max_ecc) & (new_rsq>rsq_threshold) & (np.logical_not(np.isnan(new_size))))[0]
            
            if s == 0:
                df = pd.DataFrame({'ecc': new_ecc[indices4plot],'size':new_size[indices4plot],
                                       'rsq':new_rsq[indices4plot]})
            else:
                df.append(pd.DataFrame({'ecc': new_ecc[indices4plot],'size':new_size[indices4plot],
                                       'rsq':new_rsq[indices4plot]}))
    else: 
    
        # get datapoints for RF only belonging to roi
        new_size = masked_size[roi_verts[str(roi)]]
        new_ecc = masked_eccentricity[roi_verts[str(roi)]]
        new_rsq = masked_rsq[roi_verts[str(roi)]]
        
        # define indices of voxels within region to plot
        # with rsq > 0.15, and where value not nan, ecc values between 0.25 and 4
        indices4plot = np.where((new_ecc >= min_ecc) & (new_ecc<= max_ecc) & (new_rsq>rsq_threshold) & (np.logical_not(np.isnan(new_size))))
        df = pd.DataFrame({'ecc': new_ecc[indices4plot],'size':new_size[indices4plot],
                               'rsq':new_rsq[indices4plot]})
    # sort values by eccentricity
    df = df.sort_values(by=['ecc'])  
    
    bin_size = int(len(df)/n_bins) #divide in equally sized bins
    mean_ecc = []
    mean_ecc_std = []
    mean_size = []
    mean_size_std = []
    for j in range(n_bins): # for each bin calculate rsq-weighted means and errors of binned ecc/size 
        mean_size.append(weightstats.DescrStatsW(df[bin_size*j:bin_size*(j+1)]['size'],weights=df[bin_size*j:bin_size*(j+1)]['rsq']).mean)
        mean_size_std.append(weightstats.DescrStatsW(df[bin_size*j:bin_size*(j+1)]['size'],weights=df[bin_size*j:bin_size*(j+1)]['rsq']).std_mean)
        mean_ecc.append(weightstats.DescrStatsW(df[bin_size*j:bin_size*(j+1)]['ecc'],weights=df[bin_size*j:bin_size*(j+1)]['rsq']).mean)
        mean_ecc_std.append(weightstats.DescrStatsW(df[bin_size*j:bin_size*(j+1)]['ecc'],weights=df[bin_size*j:bin_size*(j+1)]['rsq']).std_mean)
    
    if idx== 0:
        all_roi = pd.DataFrame({'mean_ecc': mean_ecc,'mean_ecc_std':mean_ecc_std,
                           'mean_size':mean_size,'mean_size_std':mean_size_std,'roi':np.tile(roi,n_bins)})
    else:
        all_roi = all_roi.append(pd.DataFrame({'mean_ecc': mean_ecc,'mean_ecc_std':mean_ecc_std,
                           'mean_size':mean_size,'mean_size_std':mean_size_std,'roi':np.tile(roi,n_bins)}),ignore_index=True)

ax = sns.lmplot(x='mean_ecc', y='mean_size', hue='roi',data=all_roi,height=8, aspect=1)
ax.set(xlabel='pRF eccentricity [dva]', ylabel='pRF size [dva]')
ax = plt.gca()
ax.axes.set_xlim(0,)
ax.axes.set_ylim(0,)
ax.set_title('ecc vs size plot, %d bins from %.2f-%.2f ecc [dva]'%(n_bins,min_ecc,max_ecc))
plt.savefig(os.path.join(figure_out,'ecc_vs_size_binned_rsq-%0.2f.svg'%rsq_threshold), dpi=100,bbox_inches = 'tight')
 

