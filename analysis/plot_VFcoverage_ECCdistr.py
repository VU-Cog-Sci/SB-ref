
# script to make visual field coverage plot with hexabins
# for different visual areas
# also test difference in ecc distributions


import os, json
import sys, glob
import numpy as np

from utils import *

import matplotlib.pyplot as plt

import cortex
import seaborn as sns
import pandas as pd

import scipy

import matplotlib.patches as patches
from statsmodels.stats import weightstats

import random

from matplotlib.lines import Line2D

import itertools


# define participant number and open json parameter file
if len(sys.argv)<2: 
    raise NameError('Please add subject number (ex:01) '    
                    'as 1st argument in the command line!') 

else:   
    sj = str(sys.argv[1]).zfill(2) #fill subject number with 0 in case user forgets 

    with open('analysis_params.json','r') as json_file: 
            analysis_params = json.load(json_file)  


# set threshold for plotting
rsq_threshold=0.17

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


if sj == 'all':
    estimates = append_pRFestimates(analysis_params['pRF_outdir'],
                                        model=fit_model,iterative=iterative_fit,exclude_subs=['sub-07'],total_chunks=total_chunks)
    
    print('appended estimates for %s excluded %s'%(str(estimates['subs']),str(estimates['exclude_subs'])))
    
    xx = np.array(estimates['x'])
    yy = np.array(estimates['y'])
       
    size = np.array(estimates['size'])
    
    beta = np.array(estimates['betas'])
    baseline = np.array(estimates['baseline'])
    
    ns = np.array(estimates['ns'])

    rsq = np.array(estimates['r2'])
    
else:
    print('loading estimates for sub-%s'%sj)
    
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


    xx = np.concatenate((estimates[0]['x'],estimates[1]['x'])); xx = xx[np.newaxis,:]
    yy = np.concatenate((estimates[0]['y'],estimates[1]['y'])); yy = yy[np.newaxis,:]
       
    size = np.concatenate((estimates[0]['size'],estimates[1]['size'])); size = size[np.newaxis,:]
    
    beta = np.concatenate((estimates[0]['betas'],estimates[1]['betas'])); beta = beta[np.newaxis,:]
    baseline = np.concatenate((estimates[0]['baseline'],estimates[1]['baseline'])); baseline = baseline[np.newaxis,:]
    
    if fit_model =='css': 
        ns = np.concatenate((estimates[0]['ns'],estimates[1]['ns'])); ns = ns[np.newaxis,:] # exponent of css
    else: #if gauss
        ns = np.ones(xx.shape); ns = ns[np.newaxis,:]

    rsq = np.concatenate((estimates[0]['r2'],estimates[1]['r2'])); rsq = rsq[np.newaxis,:]
    

masked_xx = []
masked_yy = []
masked_size = []
masked_beta = []
masked_baseline = []
masked_rsq = []
masked_ns = []
masked_polar_angle = []
masked_eccentricity = []

for w in range(xx.shape[0]): # loop once if one subject, or for all subjects when sj 'all'
    
    subject = estimates['subs'][w] if sj=='all' else 'sub-'+sj 
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
    new_estimates = mask_estimates(xx[w],yy[w],size[w],beta[w],baseline[w],rsq[w],vert_lim_dva,hor_lim_dva,ns=ns[w])

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

for idx,roi in enumerate(ROIs): #enumerate(['V1'])
    
     # get roi indices for each hemisphere
    left_roi_verts = roi_verts[roi][roi_verts[roi]<left_index]
    right_roi_verts = roi_verts[roi][roi_verts[roi]>=left_index]

    left_xx_4plot = []
    left_yy_4plot = []
    left_pa_4plot = []

    right_xx_4plot = []
    right_yy_4plot = []
    right_pa_4plot = []
    
    
    for w in range(xx.shape[0]): # loop once if one subject, or for all subjects when sj 'all'
        # LEFT HEMI
        left_xx = masked_xx[w][left_roi_verts]
        left_yy = masked_yy[w][left_roi_verts]
        left_rsq = masked_rsq[w][left_roi_verts] 
        left_pa = masked_polar_angle[w][left_roi_verts]
        
        left_xx_4plot.append(left_xx[left_rsq>rsq_threshold]) 
        left_yy_4plot.append(left_yy[left_rsq>rsq_threshold]) 
        left_pa_4plot.append(left_pa[left_rsq>rsq_threshold]) 

        # RIGHT HEMI
        right_xx = masked_xx[w][right_roi_verts]
        right_yy = masked_yy[w][right_roi_verts]
        right_rsq = masked_rsq[w][right_roi_verts] 
        right_pa = masked_polar_angle[w][right_roi_verts]
        
        right_xx_4plot.append(right_xx[right_rsq>rsq_threshold]) 
        right_yy_4plot.append(right_yy[right_rsq>rsq_threshold]) 
        right_pa_4plot.append(right_pa[right_rsq>rsq_threshold]) 
        
    f, ss = plt.subplots(1, 1, figsize=(12, 12), sharey=True)

    ss.hexbin(np.hstack(left_xx_4plot), 
              np.hstack(left_yy_4plot),
              gridsize=30, 
              cmap='Greens',
              extent= np.array([-1, 1, -1, 1]) * hor_lim_dva, #np.array([-hor_lim_dva,hor_lim_dva,-vert_lim_dva,vert_lim_dva]),
              bins='log',
              linewidths=0.0625,
              edgecolors='black',
              alpha=1)

    ss.hexbin(np.hstack(right_xx_4plot), 
              np.hstack(right_yy_4plot),
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

# make list of roi pairs, to compare distributions
ks_roi_list = np.array([pair for pair in itertools.combinations(ROIs,2)])

for _,rois_ks in enumerate(ks_roi_list):
    
    # plot Empirical distribution functions
    #rois_ks = np.array(['V1','V2']) # rois to compare ecc distribution
    
    ecc1_4plot = []
    ecc2_4plot = []
    
    for w in range(xx.shape[0]): # loop once if one subject, or for all subjects when sj 'all'
        
        new_rsq1 = masked_rsq[w][roi_verts[rois_ks[0]]]
        new_ecc1 = masked_eccentricity[w][roi_verts[rois_ks[0]]]
        indices4plot_1 = np.where((new_rsq1>rsq_threshold) & (np.logical_not(np.isnan(new_ecc1))))
        
        ecc1_4plot.append(np.sort(new_ecc1[indices4plot_1]))
        
        new_rsq2 = masked_rsq[w][roi_verts[rois_ks[1]]]
        new_ecc2 = masked_eccentricity[w][roi_verts[rois_ks[1]]]
        indices4plot_2 = np.where((new_rsq2>rsq_threshold) & (np.logical_not(np.isnan(new_ecc2))))
        
        ecc2_4plot.append(np.sort(new_ecc2[indices4plot_2]))
        

    ecc_roi1 = np.sort(np.hstack(ecc1_4plot))
    ecc_roi2 = np.sort(np.hstack(ecc2_4plot))
                              
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
    
    ecc_4plot = []
    
    for w in range(xx.shape[0]): # loop once if one subject, or for all subjects when sj 'all'
        
        new_rsq = masked_rsq[w][roi_verts[rois_ks]]
        new_ecc = masked_eccentricity[w][roi_verts[rois_ks]]
        indices4plot = np.where((new_rsq>rsq_threshold) & (np.logical_not(np.isnan(new_ecc))))
        
        ecc_4plot.append(np.sort(new_ecc[indices4plot]))

    ecc_roi = np.sort(np.hstack(ecc_4plot))
    
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
    
    ecc_4plot = []
    
    for w in range(xx.shape[0]): # loop once if one subject, or for all subjects when sj 'all'
        
        new_rsq = masked_rsq[w][roi_verts[rois_ks]]
        new_ecc = masked_eccentricity[w][roi_verts[rois_ks]]
        indices4plot = np.where((new_rsq>rsq_threshold) & (np.logical_not(np.isnan(new_ecc))))
        
        ecc_4plot.append(np.sort(new_ecc[indices4plot]))

    ecc_roi = np.sort(np.hstack(ecc_4plot))
    
    ax = sns.kdeplot(ecc_roi,label=rois_ks)
    ax.set_title('Kernel density')
    ax.set_xlabel('eccentricity [dva]')
    ax.set_ylabel('Kernel Density Estimate')
    ax.set_xlim([0,max_ecc])
    
    ax.legend()

plt.savefig(os.path.join(figure_out,'KDE_ROI-all.svg'),dpi=100)



