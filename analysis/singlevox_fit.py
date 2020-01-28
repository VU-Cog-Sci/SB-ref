
# just fit one voxel from subject
# to see differences in performance from different fit stages
# to be used in chunked data

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
    raise NameError('Please add subject number (ex: 01) '	
                    'as 1st argument in the command line!')	

elif len(sys.argv)<3:	
    raise NameError('Please add ROI name (ex: V1) or "None" if looking at voxel from no specific ROI  '	
                    'as 2nd argument in the command line!')	

elif len(sys.argv)<4:	
    raise NameError('Please voxel index number of that ROI (or from whole brain)'	
                	'as 3rd argument in the command line!')	

else:	
    sj = str(sys.argv[1]).zfill(2) #fill subject number with 0 in case user forgets	

    roi = str(sys.argv[2]) #hemifield

    index = int(sys.argv[3])	

    with open('analysis_params.json','r') as json_file:	
            analysis_params = json.load(json_file)	


# total number of chunks that were fitted (per hemi)
total_chunks = 498 #analysis_params['total_chunks']

# define paths to save plots
figure_out = os.path.join(analysis_params['derivatives'],'figures','prf','single_vox','sub-{sj}'.format(sj=sj),roi)

if not os.path.exists(figure_out): # check if path to save figures exists
    os.makedirs(figure_out) 

print('saving figures in %s'%figure_out)

## Load PRF estimates and functional data ##
median_path = os.path.join(analysis_params['pRF_outdir'],'sub-{sj}'.format(sj=sj),'run-median','chunks_'+str(total_chunks))

# load median run functional file 
med_gii = [x for x in os.listdir(median_path) if x.endswith('cropped_sg_psc.func.gii')]; med_gii.sort()

# load data for median run, one hemisphere 
hemi = ['hemi-L','hemi-R']
data = []
for _,h in enumerate(hemi):
    gii_file = os.path.join(median_path,med_gii[0]) if h == 'hemi-L' else  os.path.join(median_path,med_gii[1])
    print('using %s' %gii_file)
    data.append(np.array(surface.load_surf_data(gii_file)))

data = np.vstack(data) # will be (vertex, TR)

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

# times where bar is on screen [1st on, last on, 1st on, last on, etc] 
# not elegant, but works to define bar passing boundaries on plots
bar_onset = (np.array([14,22,25,41,55,71,74,82])-analysis_params['crop_pRF_TR'])*TR


hrf = utilities.spm_hrf(0,TR)

# make stimulus object, which takes an input design matrix and sets up its real-world dimensions
prf_stim = PRFStimulus2D(screen_size_cm=analysis_params["screen_width"], 
                         screen_distance_cm=analysis_params["screen_distance"], 
                         design_matrix=prf_dm, 
                         TR=TR)

# get vertices from ROI for subject fsaverage
if roi != 'None':
    roi_verts = cortex.get_roi_verts('fsaverage_gross',roi)[roi]
    data = data[roi_verts] # data from ROI


# models to get single voxel and compare fits
models = ['grid','it_gauss','it_css']

for _,model in enumerate(models):

    if model=='grid':
        estimates_list = [x for x in os.listdir(median_path) if 'iterative' not in x and 'chunk' not in x and x.endswith('gauss_estimates.npz')]
        if not estimates_list: #list is empty
            # need to combine chunks and make combined estimates array
            estimates_list = join_chunks(estimates_list,median_path,chunk_num=total_chunks,fit_model='gauss')

    elif model=='it_gauss':
        estimates_list = [x for x in os.listdir(median_path) if 'iterative' in x and 'chunk' not in x and x.endswith('gauss_estimates.npz')]
        if not estimates_list: #list is empty
            # need to combine chunks and make combined estimates array
            estimates_list = join_chunks(estimates_list,median_path,chunk_num=total_chunks,fit_model='gauss')

    else: # iterative css
        estimates_list = [x for x in os.listdir(median_path) if 'iterative' in x and 'chunk' not in x and x.endswith('css_estimates.npz')]
        if not estimates_list: #list is empty
            # need to combine chunks and make combined estimates array
            estimates_list = join_chunks(estimates_list,median_path,chunk_num=total_chunks,fit_model='css')

    estimates_list.sort() #sort to make sure pRFs not flipped

    estimates = []
    for _,val in enumerate(estimates_list) :
        print('appending %s'%val)
        estimates.append(np.load(os.path.join(median_path, val))) #save both hemisphere estimates in same array


    xx = np.concatenate((estimates[0]['x'],estimates[1]['x']))
    yy = np.concatenate((estimates[0]['y'],estimates[1]['y']))
       
    size = np.concatenate((estimates[0]['size'],estimates[1]['size']))
    
    beta = np.concatenate((estimates[0]['betas'],estimates[1]['betas']))
    baseline = np.concatenate((estimates[0]['baseline'],estimates[1]['baseline']))
    
    if model =='it_css': 
        ns = np.concatenate((estimates[0]['ns'],estimates[1]['ns'])) # exponent of css
    else: #if gauss
        ns = np.ones(xx.shape)

    rsq = np.concatenate((estimates[0]['r2'],estimates[1]['r2']))

    # set limits for xx and yy, forcing it to be within the screen boundaries

    vert_lim_dva = (analysis_params['screenRes'][-1]/2) * dva_per_pix(analysis_params['screen_width'],analysis_params['screen_distance'],analysis_params['screenRes'][-1])
    hor_lim_dva = (analysis_params['screenRes'][0]/2) * dva_per_pix(analysis_params['screen_width'],analysis_params['screen_distance'],analysis_params['screenRes'][-1])

    # make new variables that are masked (center of RF within screen limits and only positive pRFs)
    # also max size of RF is 10 dva
    print('masking variables to be within screen and only show positive RF')
    new_estimates = mask_estimates(xx,yy,size,beta,baseline,rsq,vert_lim_dva,hor_lim_dva,ns=ns)

    masked_xx = new_estimates['x']
    masked_yy = new_estimates['y']
    masked_size = new_estimates['size']
    masked_beta = new_estimates['beta']
    masked_baseline = new_estimates['baseline']
    masked_rsq = new_estimates['rsq']
    masked_ns = new_estimates['ns']

    if roi != 'None':
        new_rsq = masked_rsq[roi_verts]

        new_xx = masked_xx[roi_verts]
        new_yy = masked_yy[roi_verts]
        new_size = masked_size[roi_verts]

        new_beta = masked_beta[roi_verts]
        new_baseline = masked_baseline[roi_verts]
        
        new_ns = masked_ns[roi_verts]

    else:
        new_rsq = masked_rsq

        new_xx = masked_xx
        new_yy = masked_yy
        new_size = masked_size

        new_beta = masked_beta
        new_baseline = masked_baseline
        
        new_ns = masked_ns


    if model == 'it_css':
        gg = CSS_Iso2DGaussianGridder(stimulus=prf_stim,
                                      hrf=hrf,
                                      filter_predictions=True,
                                      window_length=analysis_params["sg_filt_window_length"],
                                      polyorder=analysis_params["sg_filt_polyorder"],
                                      highpass=True,
                                      task_lengths=np.array([prf_dm.shape[-1]]))

        model_fit = gg.return_single_prediction(new_xx[index],new_yy[index],new_size[index], 
                                                         new_beta[index],new_baseline[index],new_ns[index])

    else:
        gg = Iso2DGaussianGridder(stimulus=prf_stim,
                                  hrf=hrf,
                                  filter_predictions=True,
                                  window_length=analysis_params["sg_filt_window_length"],
                                  polyorder=analysis_params["sg_filt_polyorder"],
                                  highpass=True,
                                  task_lengths=np.array([prf_dm.shape[-1]]))

        model_fit = gg.return_single_prediction(new_xx[index],new_yy[index],new_size[index], 
                                                         new_beta[index],new_baseline[index])


    timeseries = data[index]

    # plot data with model
    time_sec = np.linspace(0,len(timeseries)*TR,num=len(timeseries)) # array with 90 timepoints, in seconds
    
    fig, ax = plt.subplots(1,figsize=(15,7.5),dpi=100)
    ax.plot(time_sec,model_fit,c='#db3050',lw=3,label='prf model',zorder=1)
    ax.scatter(time_sec,timeseries, marker='v',c='k',label='data')
    ax.set_xlabel('Time (s)',fontsize=18)
    ax.set_ylabel('BOLD signal change (%)',fontsize=18)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlim(0,len(timeseries)*TR)

    # plot axis vertical bar on background to indicate stimulus display time
    ax_count = 0
    for h in range(4):
        plt.axvspan(bar_onset[ax_count], bar_onset[ax_count+1]+TR, facecolor='#db3050', alpha=0.1)
        ax_count += 2

    plt.title('%s fit - x: %.2f, y: %.2f, size: %.2f, beta: %.2f, base: %.2f, n: %.2f, r2: %.2f ' %(model,
                                                                                                    new_xx[index],
                                                                                                    new_yy[index],
                                                                                                    new_size[index],
                                                                                                    new_beta[index],
                                                                                                    new_baseline[index],
                                                                                                    new_ns[index],
                                                                                                    new_rsq[index]))

    fig.savefig(os.path.join(figure_out,'pRF_singvoxfit_timeseries_ROI-%s_index-%d_fit-%s.png'%(roi,index,model)), dpi=100,bbox_inches = 'tight')

   
                





        



    

























