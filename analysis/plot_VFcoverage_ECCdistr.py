
# script to make visual field coverage plot with hexabins
# for different visual areas
# also test difference in ecc distributions


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

import matplotlib.patches as patches
from statsmodels.stats import weightstats

import random

from matplotlib.lines import Line2D


# define participant number and open json parameter file
if len(sys.argv)<2: 
    raise NameError('Please add subject number (ex:01) '    
                    'as 1st argument in the command line!') 

else:   
    sj = str(sys.argv[1]).zfill(2) #fill subject number with 0 in case user forgets 

    with open('analysis_params.json','r') as json_file: 
            analysis_params = json.load(json_file)  


# fit model to use (gauss or css)
fit_model = 'css' #analysis_params["fit_model"]
# if using estimates from iterative fit
iterative_fit = True #True

# total number of chunks that were fitted (per hemi)
total_chunks = analysis_params['total_chunks']

# define paths to save plots
figure_out = os.path.join(analysis_params['derivatives'],'VF_coverage','sub-{sj}'.format(sj=sj))
    
if not os.path.exists(figure_out): # check if path to save figures exists
    os.makedirs(figure_out) 

print('saving figures in %s'%figure_out)


if sj != 'all': # inidividual subjects
    
    median_path = os.path.join(analysis_params['pRF_outdir'],'sub-{sj}'.format(sj=sj),'run-median','chunks_'+str(total_chunks))
    
    if iterative_fit==True: # if selecting params from iterative fit
        estimates_list = [x for x in os.listdir(median_path) if 'iterative' in x and 'chunk' not in x and x.endswith(fit_model+'_estimates.npz')]
        if not estimates_list: #list is empty
            # combine chunks and get new estimates list
            list_all =  [x for x in os.listdir(median_path) if 'iterative' in x and 'chunk' in x and x.endswith(fit_model+'_estimates.npz')]
            estimates_list = join_chunks(list_all,median_path,chunk_num=total_chunks,fit_model=fit_model) # combine chunks and get new estimates list 
    else: # only look at grid fit
        estimates_list = [x for x in os.listdir(median_path) if 'iterative' not in x and 'chunk' not in x and x.endswith(fit_model+'_estimates.npz')]
        if not estimates_list: #list is empty
            # combine chunks and get new estimates list
            list_all =  [x for x in os.listdir(median_path) if 'iterative' not in x and 'chunk' in x and x.endswith(fit_model+'_estimates.npz')]
            estimates_list = join_chunks(list_all,median_path,chunk_num=total_chunks,fit_model=fit_model) # combine chunks and get new estimates list 

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
    
    if fit_model =='css': 
        ns = np.concatenate((estimates[0]['ns'],estimates[1]['ns'])) # exponent of css
    else: #if gauss
        ns = np.ones(xx.shape)

    rsq = np.concatenate((estimates[0]['r2'],estimates[1]['r2']))


# set limits for xx and yy, forcing it to be within the screen boundaries
if sj in ['02','11','12','13']: # linux computer has different res

    vert_lim_dva = (analysis_params['screenRes_HD'][-1]/2) * dva_per_pix(analysis_params['screen_width'],analysis_params['screen_distance'],analysis_params['screenRes_HD'][0])
    hor_lim_dva = (analysis_params['screenRes_HD'][0]/2) * dva_per_pix(analysis_params['screen_width'],analysis_params['screen_distance'],analysis_params['screenRes_HD'][0])


else:    
    vert_lim_dva = (analysis_params['screenRes'][-1]/2) * dva_per_pix(analysis_params['screen_width'],analysis_params['screen_distance'],analysis_params['screenRes'][0])
    hor_lim_dva = (analysis_params['screenRes'][0]/2) * dva_per_pix(analysis_params['screen_width'],analysis_params['screen_distance'],analysis_params['screenRes'][0])

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

# set threshold for plotting
rsq_threshold=0.1

# now construct polar angle and eccentricity values
complex_location = masked_xx + masked_yy * 1j
masked_polar_angle = np.angle(complex_location)
masked_eccentricity = np.abs(complex_location)


# get vertices for subject fsaverage
# for later plotting

ROIs = ['V1','V2','V3','sPCS','iPCS'] # list will combine those areas, string only accounts for 1 ROI

roi_verts = {} #empty dictionary 
for i,val in enumerate(ROIs):
    
    if type(val)==str: # if string, we can directly get the ROI vertices  
        roi_verts[val] = cortex.get_roi_verts('fsaverage_meridians',val)[val]

    else: # if list
        indice = []
        for w in range(len(ROIs[0])): # load vertices for each region of list
            indice.append(cortex.get_roi_verts('fsaverage_meridians',val[w])[val[w]])
        
        roi_verts[str(val)] = np.hstack(indice)


# get mid vertex index (diving hemispheres)
left_index = cortex.db.get_surfinfo('fsaverage').left.shape[0] 

# coverage plots - full VF 

for idx,roi in enumerate(ROIs):#enumerate(['V1']): #enumerate(ROIs):
    
     # get roi indices for each hemisphere
    left_roi_verts = roi_verts[roi][roi_verts[roi]<left_index]
    right_roi_verts = roi_verts[roi][roi_verts[roi]>=left_index]

    # LEFT HEMI
    left_xx = masked_xx[left_roi_verts]
    left_yy = masked_yy[left_roi_verts]
    left_rsq = masked_rsq[left_roi_verts]
    
    # now construct polar angle 
    left_complex_rf = left_xx + left_yy * 1j
    left_pa = np.angle(left_complex_rf)

    # RIGHT HEMI
    right_xx = masked_xx[right_roi_verts]
    right_yy = masked_yy[right_roi_verts]
    right_rsq = masked_rsq[right_roi_verts]
    
    # now construct polar angle 
    right_complex_rf = right_xx + right_yy * 1j
    right_pa = np.angle(right_complex_rf)
    
    f, ss = plt.subplots(1, 1, figsize=(12, 12), sharey=True)

    ss.hexbin(left_xx[left_rsq>rsq_threshold], 
              left_yy[left_rsq>rsq_threshold],
              #C=left_rsq[left_rsq>rsq_threshold],
              #reduce_C_function=np.sum,
              gridsize=30, 
              cmap='Greens',
              extent= np.array([-1, 1, -1, 1]) * hor_lim_dva, #np.array([-hor_lim_dva,hor_lim_dva,-vert_lim_dva,vert_lim_dva]),
              bins='log',
              linewidths=0.0625,
              edgecolors='black',
              alpha=1)

    ss.hexbin(right_xx[right_rsq>rsq_threshold], 
              right_yy[right_rsq>rsq_threshold],
              #C=right_rsq[right_rsq>rsq_threshold],
              #reduce_C_function=np.sum,
              gridsize=30, 
              cmap='Reds', #'YlOrRd_r',
              extent= np.array([-1, 1, -1, 1]) * hor_lim_dva, #np.array([-hor_lim_dva,hor_lim_dva,-vert_lim_dva,vert_lim_dva]),
              bins='log',
              linewidths=0.0625,
              edgecolors='black',
              alpha=0.5)

    ss.set_title('Visual field coverage for %s'%roi)
    ss.set_xlabel('Horizontal visual position [dva]')
    ss.set_ylabel('Vertical visual position [dva]')
    plt.tight_layout()
    # set middle lines
    ss.axvline(0, -hor_lim_dva, hor_lim_dva, lw=0.25, color='w')
    ss.axhline(0, -hor_lim_dva, hor_lim_dva, lw=0.25, color='w')
    
    # custom lines only to make labels
    custom_lines = [Line2D([0], [0], color='g',alpha=0.5, lw=4),
                    Line2D([0], [0], color='r',alpha=0.5, lw=4)]

    plt.legend(custom_lines, ['LH', 'RH'])

    #sns.despine(f, offset=10)

    # Create a Rectangle patch
    #rect = patches.Rectangle((-hor_lim_dva,-vert_lim_dva),hor_lim_dva*2,vert_lim_dva*2,linewidth=1,linestyle='--',edgecolor='w',facecolor='none',zorder=10)
    # Add the patch to the Axes
    #ss.add_patch(rect) 
    plt.savefig(os.path.join(figure_out,'VF_coverage_ROI-%s_hemi-both.svg'%roi),dpi=100)
    
    f, ss = plt.subplots(1, 1, figsize=(12, 12), sharey=True)
    plt.hist(left_pa,color='g',alpha=0.5,label='LH')
    plt.hist(right_pa,color='r',alpha=0.5,label='RH')
    plt.xlabel('Polar angle')
    plt.legend()
    plt.title('Histogram of PA distribution for %s'%roi)
    plt.savefig(os.path.join(figure_out,'PA_histogram_ROI-%s_hemi-both.svg'%roi),dpi=100)

 
# Check ECC distributions

# max eccentricity within screen dva
max_ecc = np.sqrt(vert_lim_dva**2 + hor_lim_dva**2)

ks_roi_list = [['V1','V2'],['V1','V3'],['V2','V3'],['V1','sPCS'],['V1','iPCS'],['sPCS','iPCS']]

for _,rois_ks in enumerate(ks_roi_list):
    
    # plot Empirical distribution functions
    #rois_ks = np.array(['V1','V2']) # rois to compare ecc distribution
    
    new_rsq1 = masked_rsq[roi_verts[rois_ks[0]]]
    new_ecc1 = masked_eccentricity[roi_verts[rois_ks[0]]]
    indices4plot_1 = np.where((new_rsq1>rsq_threshold) & (np.logical_not(np.isnan(new_ecc1))))
    
    new_rsq2 = masked_rsq[roi_verts[rois_ks[1]]]
    new_ecc2 = masked_eccentricity[roi_verts[rois_ks[1]]]
    indices4plot_2 = np.where((new_rsq2>rsq_threshold) & (np.logical_not(np.isnan(new_ecc2))))

    ecc_roi1 = np.sort(new_ecc1[indices4plot_1])
    ecc_roi2 = np.sort(new_ecc2[indices4plot_2])
                              
    fig, ax = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

    n1 = np.arange(1,len(ecc_roi1)+1) / np.float(len(ecc_roi1))
    ax[0].step(ecc_roi1,n1,color='r',label=rois_ks[0])

    n2 = np.arange(1,len(ecc_roi2)+1) / np.float(len(ecc_roi2))
    ax[0].step(ecc_roi2,n2,color='b',label=rois_ks[1])

    ax[0].legend()

    ax[0].set_title('Empirical cumulative distribution %s, KS test p-value = %.3f'%(str(rois_ks),scipy.stats.ks_2samp(ecc_roi1, ecc_roi2)[1]))
    ax[0].set_xlabel('eccentricity [dva]')
    ax[0].set_ylabel('Cumulative probability')
    ax[0].set_xlim([0,max_ecc])
    ax[0].set_ylim([0,1])
    
    ax[1] = sns.kdeplot(ecc_roi1,label=rois_ks[0])
    ax[1] = sns.kdeplot(ecc_roi2,label=rois_ks[1])
    ax[1].set_title('Kernel density %s'%(str(rois_ks)))
    ax[1].set_xlabel('eccentricity [dva]')
    ax[1].set_ylabel('Kernel Density Estimate')
    ax[1].set_xlim([0,max_ecc])
    #ax[1].set_ylim([0,0.4])
    
    plt.savefig(os.path.join(figure_out,'ECDF_ROI-%s.svg'%str(rois_ks)),dpi=100)


# All roi combined

fig, ax = plt.subplots(1, 1, figsize=(16, 8), sharey=True)

for _,rois_ks in enumerate(ROIs):
    
    # plot Empirical distribution functions
    #rois_ks = np.array(['V1','V2']) # rois to compare ecc distribution
    new_rsq = masked_rsq[roi_verts[rois_ks]]
    new_ecc = masked_eccentricity[roi_verts[rois_ks]]
    indices4plot = np.where((new_rsq>rsq_threshold) & (np.logical_not(np.isnan(new_ecc))))

    ecc_roi = np.sort(new_ecc[indices4plot])

    n1 = np.arange(1,len(ecc_roi)+1) / np.float(len(ecc_roi))
    ax.step(ecc_roi,n1,label=rois_ks)

    ax.legend()

    ax.set_title('Empirical cumulative distribution')
    ax.set_xlabel('eccentricity [dva]')
    ax.set_ylabel('Cumulative probability')
    ax.set_xlim([0,max_ecc])
    ax.set_ylim([0,1])

plt.savefig(os.path.join(figure_out,'ECDF_ROI-all.svg'),dpi=100)


fig, ax = plt.subplots(1, 1, figsize=(16, 8), sharey=True)

for _,rois_ks in enumerate(ROIs):
    
    # plot Empirical distribution functions
    #rois_ks = np.array(['V1','V2']) # rois to compare ecc distribution
    new_rsq = masked_rsq[roi_verts[rois_ks]]
    new_ecc = masked_eccentricity[roi_verts[rois_ks]]
    indices4plot = np.where((new_rsq>rsq_threshold) & (np.logical_not(np.isnan(new_ecc))))

    ecc_roi = np.sort(new_ecc[indices4plot])
    
    ax = sns.kdeplot(ecc_roi,label=rois_ks)
    ax.set_title('Kernel density')
    ax.set_xlabel('eccentricity [dva]')
    ax.set_ylabel('Kernel Density Estimate')
    ax.set_xlim([0,max_ecc])
    
    ax.legend()

plt.savefig(os.path.join(figure_out,'KDE_ROI-all.svg'),dpi=100)
