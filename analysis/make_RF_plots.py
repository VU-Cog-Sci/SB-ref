
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


# define participant number and open json parameter file
if len(sys.argv)<2:	
    raise NameError('Please add subject number (ex:01) '	
                    'as 1st argument in the command line!')	

else:	
    sj = str(sys.argv[1]).zfill(2) #fill subject number with 0 in case user forgets	

    with open('analysis_params.json','r') as json_file:	
            analysis_params = json.load(json_file)	


# define paths
figure_out = os.path.join(analysis_params['derivatives'],'figures','prf','sub-{sj}'.format(sj=sj))

if not os.path.exists(figure_out): # check if path to save figures exists
    os.makedirs(figure_out) 

with_smooth = 'False'#analysis_params['with_smooth']

## Load PRF estimates ##
if sj=='median':
    estimates = median_iterative_pRFestimates(analysis_params['pRF_outdir'],with_smooth=bool(strtobool(with_smooth)))
    print('computed median estimates for %s'%str(estimates['subs']))
    xx = estimates['x']
    yy = estimates['y']
    rsq = estimates['r2']
    size = estimates['size']
    beta = estimates['betas']
    baseline = estimates['baseline']
    
else:
    if with_smooth=='True':    
        median_path = os.path.join(analysis_params['pRF_outdir'],'sub-{sj}'.format(sj=sj),'run-median','smooth%d'%analysis_params['smooth_fwhm'],'iterative_fit')
    else:
        median_path = os.path.join(analysis_params['pRF_outdir'],'sub-{sj}'.format(sj=sj),'run-median','iterative_fit')

    estimates_list = [x for x in os.listdir(median_path) if x.endswith('iterative_output.npz')]
    estimates_list.sort() #sort to make sure pRFs not flipped

    estimates = []
    for _,val in enumerate(estimates_list) :
        print('appending %s'%val)
        estimates.append(np.load(os.path.join(median_path, val))) #save both hemisphere estimates in same array
        
    xx = np.concatenate((estimates[0]['it_output'][...,0],estimates[1]['it_output'][...,0]))
    yy = np.concatenate((estimates[0]['it_output'][...,1],estimates[1]['it_output'][...,1]))
    yy = -yy # Need to do this for now, CHANGE ONCE BUG FIXED

    size = np.concatenate((estimates[0]['it_output'][...,2],estimates[1]['it_output'][...,2]))
    beta = np.concatenate((estimates[0]['it_output'][...,3],estimates[1]['it_output'][...,3]))
    baseline = np.concatenate((estimates[0]['it_output'][...,4],estimates[1]['it_output'][...,4]))

    rsq = np.concatenate((estimates[0]['it_output'][...,5],estimates[1]['it_output'][...,5])) 


# now construct polar angle and eccentricity values
rsq_threshold = 0.2#analysis_params['rsq_threshold']

complex_location = xx + yy * 1j
polar_angle = np.angle(complex_location)
eccentricity = np.abs(complex_location)

# normalize polar angles to have values in circle between 0 and 1
polar_ang_norm = (polar_angle + np.pi) / (np.pi * 2.0)

# use "resto da divisão" so that 1 == 0 (because they overlapp in circle)
# why have an offset?
angle_offset = 0.85#0.1
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
alpha[alpha_mask] = np.nan


images = {}

images['polar'] = cortex.VertexRGB(rgb[..., 0].T, 
                                 rgb[..., 1].T, 
                                 rgb[..., 2].T, 
                                 subject='fsaverage', alpha=alpha)
#cortex.quickshow(images['polar'],with_curvature=True,with_sulci=True,with_colorbar=False)
filename = os.path.join(figure_out,'flatmap_space-fsaverage_rsq-%0.2f_type-polar_angle.svg' %rsq_threshold)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['polar'], recache=True,with_colorbar=False,with_curvature=True,with_sulci=True)

# vertex for rsq
images['rsq'] = cortex.Vertex2D(rsq.T, alpha, 'fsaverage',
                           vmin=0, vmax=1.0,
                           vmin2=0, vmax2=1.0, cmap='Reds_cov')
#cortex.quickshow(images['rsq'],with_curvature=True,with_sulci=True)
filename = os.path.join(figure_out,'flatmap_space-fsaverage_rsq-%0.2f_type-rsquared.svg' %rsq_threshold)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['rsq'], recache=True,with_colorbar=True,with_curvature=True,with_sulci=True)


# vertex for ecc
ecc4plot = np.zeros(eccentricity.shape); ecc4plot[:]=np.nan
ecc4plot[rsq>=rsq_threshold]= eccentricity[rsq>=rsq_threshold]


images['ecc'] = cortex.Vertex(ecc4plot.T, 'fsaverage',
                           vmin=0, vmax=analysis_params['max_eccen'],
                           cmap='J4')
#cortex.quickshow(images['ecc'],with_curvature=True,with_sulci=True)
# Save this flatmap
filename = os.path.join(figure_out,'flatmap_space-fsaverage_rsq-%0.2f_type-eccentricity.svg' %rsq_threshold)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['ecc'], recache=True,with_colorbar=True,with_curvature=True,with_sulci=True)


# vertex for ecc
size4plot = np.zeros(size.shape); size4plot[:]=np.nan
size4plot[rsq>=rsq_threshold]= size[rsq>=rsq_threshold]


# vertex for size
images['size'] = cortex.dataset.Vertex(size4plot.T, 'fsaverage',
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
        roi_verts[val] = cortex.get_roi_verts('fsaverage',val)[val]

    else: # if list
        indice = []
        for w in range(len(ROIs[0])): # load vertices for each region of list
            indice.append(cortex.get_roi_verts('fsaverage',val[w])[val[w]])
        
        roi_verts[str(val)] = np.hstack(indice)


# plot
for idx,roi in enumerate(ROIs):
    
    f, s = plt.subplots(1, 2, figsize=(24,48))

    if type(roi)!=str: # if list
        roi = str(roi)
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
    
    #####
    ## normalize polar angles to have values in circle between 0 and 1
    #new_pa_norm = (new_polar_angle + np.pi) / (np.pi * 2.0)
    #
    ## use "resto da divisão" so that 1 == 0 (because they overlapp in circle)
    ## why have an offset?
    #new_pa_norm = np.fmod(new_pa_norm+angle_offset, 1.0)
    #
    ## convert angles to colors, using correlations as weights
    #hsv_colors = np.zeros(list(new_pa_norm.shape) + [3])
    #hsv_colors[..., 0] = new_pa_norm # different hue value for each angle
    #hsv_colors[..., 1] = new_rsq/max(new_rsq)#new_rsq #1 #saturation weighted by rsq
    #hsv_colors[..., 2] = 1#(new_rsq > rsq_threshold).astype(float) # value weighted by rsq
    #
    ## convert hsv values of np array to rgb values (values assumed to be in range [0, 1])
    #rgba_colors = colors.hsv_to_rgb(hsv_colors)
    ######## 

    edgecolors = np.zeros((new_xx.shape[0], 4))
    edgecolors[:,:3] = 1#0.8 # make gray
    edgecolors[:,3] = new_rsq/10 #12# the rsq is alpha
    
    s[0].scatter(new_xx, new_yy, s=(4*np.pi*new_size)**2, color=rgba_colors, edgecolors=edgecolors, linewidths=2) # this size is made up and depends on dpi - beware.
    s[0].set_title('%s pRFs in visual field'%roi)
    s[0].set_xlim([-analysis_params["max_eccen"],analysis_params["max_eccen"]])
    s[0].set_ylim([-analysis_params["max_eccen"],analysis_params["max_eccen"]])
    s[0].axvline(0, -15, 15, c='k', lw=0.25)
    s[0].axhline(0, -15, 15, c='k', lw=0.25)
    s[0].set_xlabel('horizontal space [dva]')
    s[0].set_ylabel('vertical space [dva]')
    s[1].set_title('%s pRF size vs eccentricity'%roi)
    s[1].set_xlim([0,15])
    s[1].set_ylim([0,analysis_params["max_size"]])
    s[1].set_xlabel('pRF eccentricity [dva]')
    s[1].set_ylabel('pRF size [dva]')
    s[1].scatter(new_ecc, new_size, color=rgba_colors, edgecolors=edgecolors, linewidths=2);  # this size is made up - beware.
    
    # make sure that plots have proportional size
    s[0].set(adjustable='box-forced', aspect='equal')
    s[1].set(adjustable='box-forced', aspect='equal')
    f.savefig(os.path.join(figure_out,'RF_scatter_ROI-%s.svg'%roi), dpi=100,bbox_inches = 'tight')
    
#plt.show() 


# Do similar plots, but combined in same image
# using polar angle as hue (more for checking if everything coeherent)

# plot
f, s = plt.subplots(len(ROIs), 2, figsize=(24,48))
for idx,roi in enumerate(ROIs):
    if type(roi)!=str: # if list
        roi = str(roi)
    # get datapoints for RF only belonging to roi
    new_rsq = rsq[roi_verts[roi]]

    new_xx = xx[roi_verts[roi]]
    new_yy = yy[roi_verts[roi]]
    new_size = size[roi_verts[roi]]
    
    new_ecc = eccentricity[roi_verts[roi]]
    new_polar_angle = polar_angle[roi_verts[roi]]
    
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
    rgba_colors = colors.hsv_to_rgb(hsv_colors)
    ######## 

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
    
    # make sure that plots have proportional size
    s[idx][0].set(adjustable='box-forced', aspect='equal')
    s[idx][1].set(adjustable='box-forced', aspect='equal')

f.savefig(os.path.join(figure_out,'RF_pa_scatter_allROIs.png'), dpi=100,bbox_inches = 'tight')
#plt.show() 

# now do single voxel fits, choosing voxels with highest rsq 
# from each ROI (early visual vs sPCS vs iPCS)
# and plot all in same figure (usefull for fig1)

if sj != 'median': # doesn't work for median subject
    # path to functional files
    filepath = glob.glob(os.path.join(analysis_params['post_fmriprep_outdir'], 'prf', 'sub-{sj}'.format(sj=sj), '*'))
    print('functional files from %s' % os.path.split(filepath[0])[0])

    # last part of filename to use
    file_extension = '_sg_psc.func.gii'

    # list of functional files (5 runs)
    filename = [run for run in filepath if 'prf' in run and 'fsaverage' in run and run.endswith(file_extension)]
    filename.sort()

    # path to save fits, for testing
    out_dir = os.path.join(analysis_params['pRF_outdir'],'sub-{sj}'.format(sj=sj),'tests')

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
                              highpass=True)



    ##### #################################### #############
    
    ############ SHOW FIT FOR SINGLE VOXEL AND PLOT IT OVER DATA ##############

    # get single voxel data from ROI where rsq is max 
    # get datapoints for RF only belonging to roi
    for idx,roi in enumerate(ROIs):
        if type(roi)!=str: # if list
            print('skipping list ROI %s for timeseries plot'%str(roi))
        else:
            new_rsq = rsq[roi_verts[roi]]

            new_xx = xx[roi_verts[roi]]
            new_yy = yy[roi_verts[roi]]
            new_size = size[roi_verts[roi]]

            new_beta = beta[roi_verts[roi]]
            new_baseline = baseline[roi_verts[roi]]

            new_ecc = eccentricity[roi_verts[roi]]
            new_polar_angle = polar_angle[roi_verts[roi]]


            new_data = data[roi_verts[roi]] # data from ROI

            new_index =np.where(new_rsq==max(new_rsq))[0][0]# index for max rsq within ROI

            timeseries = new_data[new_index]

            model_it_prfpy = gg.return_single_prediction(new_xx[new_index],new_yy[new_index],new_size[new_index],
                                                         beta=new_beta[new_index],baseline=new_baseline[new_index])
            model_it_prfpy = model_it_prfpy[0]

            print('voxel %d of ROI %s , rsq of fit is %.3f' %(new_index,roi,new_rsq[new_index]))

            # plot data with model
            time_sec = np.linspace(0,len(timeseries)*TR,num=len(timeseries)) # array with 90 timepoints, in seconds
            fig= plt.figure(figsize=(15,7.5),dpi=100)
            plt.plot(time_sec,model_it_prfpy,c='r',lw=3,label='model fit',zorder=1)
            plt.scatter(time_sec,timeseries, marker='.',c='k',label='data')
            plt.xlabel('Time (s)',fontsize=18)
            plt.ylabel('BOLD signal change (%)',fontsize=18)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.xlim(0,len(timeseries)*TR)
            plt.title('voxel %d of ROI %s , rsq of fit is %.3f' %(new_index,roi,new_rsq[new_index]))
            plt.legend(loc=0)
            fig.savefig(os.path.join(figure_out,'pRF_singvoxfit_timeseries_%s.svg'%roi), dpi=100,bbox_inches = 'tight')

            # get receptive field for that voxel
            rf_prfpy = gauss2D_iso_cart(x=gg.stimulus.x_coordinates[..., np.newaxis],
                                        y=gg.stimulus.y_coordinates[..., np.newaxis],
                                        mu=(new_xx[new_index], new_yy[new_index]),
                                        sigma=new_size[new_index])

            # plot receptive field
            fig= plt.figure(figsize=(15,7.5))
            if os.path.split(dm_filename)[-1] == 'prf_dm_square.npy': # if square DM
                rf2plot = rf_prfpy[:,31:136,0] # cut upper and lower borders, good enough for plotting purposes
            else:
                rf2plot = rf_prfpy[:,:,0]

            plt.imshow(rf2plot.T,cmap='viridis')
            plt.title('RF for single voxel %d of ROI %s ' %(new_index,roi))
            plt.axis('off')
            fig.savefig(os.path.join(figure_out,'RF_singvoxfit_%s.svg'%roi), dpi=100,bbox_inches = 'tight')


    # redo vertice list (repetitive but quick fix for now)
    ROIs = [['V1','V2','V3'],'sPCS','iPCS']
    
    # Plot median timeseries, just to check for bias
    fig= plt.figure(figsize=(15,7.5))

    for i in range(len(ROIs)):
        c = ['r','g','b']
        new_data = data[roi_verts[str(ROIs[i])]] # data from ROI

        plt.plot(time_sec,np.nanmedian(new_data[rsq[roi_verts[str(ROIs[i])]]>0.2],axis=0),c=c[i],label=str(ROIs[i]))
        plt.legend(loc=0)
    plt.xlabel('Time (s)',fontsize=18)
    plt.ylabel('BOLD signal change (%)',fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(0,90*TR)
    plt.show()    
    fig.savefig(os.path.join(figure_out,'pRF_median_timeseries_allROIs.svg'), dpi=100,bbox_inches = 'tight')

    
# make plot to show relationship between ecc and size for different regions
# redo vertice list (repetitive but quick fix for now)
ROIs = [['V1','V2','V3'],'sPCS','iPCS']
# bin the data   
num_bins = 10 # number of bins
bin_array = np.linspace(0,16,num=num_bins+1) # array that goes from 2 to 2, then bin will be values between 0-2, 2-4 etc

f, s = plt.subplots(1, 1, figsize=(12,6))
c = ['r','g','b']
for idx,roi in enumerate(ROIs):
    # get datapoints for RF only belonging to roi
    new_size = size[roi_verts[str(roi)]]
    new_ecc = eccentricity[roi_verts[str(roi)]]

    size_binned = []
    std_bin = []
    for i in range(num_bins): # append median size value for bin
        size_binned.append(np.nanmedian(new_size[np.where((new_ecc>bin_array[i])&(new_ecc<bin_array[i+1]))[0]]))
        std_bin.append(np.std(new_size[np.where((new_ecc>bin_array[i])&(new_ecc<bin_array[i+1]))[0]]))

    z = np.polyfit(range(num_bins), size_binned, 1)
    p = np.poly1d(z)
    plt.scatter(range(num_bins),size_binned,c=c[idx])
    #plt.errorbar(range(num_bins),size_binned,yerr=std_bin, linestyle="None",c=c[idx])
    plt.plot(range(num_bins),p(range(num_bins)),color=c[idx],linestyle='--',label=str(str(roi)))

plt.legend(loc=0)
plt.xlabel('pRF eccentricity [bins]',fontsize=18)
plt.ylabel('pRF size [dva]',fontsize=18)
s.set(adjustable='box-forced', aspect='equal')    
plt.show()   
f.savefig(os.path.join(figure_out,'ecc_vs_size_binned.svg'), dpi=100,bbox_inches = 'tight')













