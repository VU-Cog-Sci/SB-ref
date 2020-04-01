
# make retino plots
# which will take as input subject number (01 or median)
# and produce relevant plots to check retinotopy
# save plots in figures/prf/fig_final folder within derivatives


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
import matplotlib.gridspec as gridspec

import random

from matplotlib.lines import Line2D

import itertools

# requires pfpy be installed - preferably with python setup.py develop
from prfpy.rf import *
from prfpy.timecourse import *
from prfpy.stimulus import PRFStimulus2D
from prfpy.grid import Iso2DGaussianGridder,CSS_Iso2DGaussianGridder
from prfpy.fit import Iso2DGaussianFitter, CSS_Iso2DGaussianFitter

from popeye import utilities 


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
rsq_threshold=0.17


if iterative_fit==True:
    figure_out = os.path.join(figure_out,'iterative','sub-{sj}'.format(sj=sj))
else:
    figure_out = os.path.join(figure_out,'grid','sub-{sj}'.format(sj=sj))
        
if not os.path.exists(figure_out): # check if path to save figures exists
    os.makedirs(figure_out) 

print('saving figures in %s'%figure_out)

# make list with subjects to append and use (or not)
if sj == 'median':
    excl_subs = ['sub-07','sub-03','sub-13']
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
    

# get vertices for subject fsaverage
# for later plotting

ROIs = ['V1','V2','V3','V3AB','hV4','LO','IPS0','IPS1','IPS2+','sPCS','iPCS'] # list will combine those areas, string only accounts for 1 ROI

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


####### coverage plots - full VF ###########

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
    
    
    for w in range(len(estimates['subs'])): # loop once if one subject, or for all subjects when sj 'median'
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
              gridsize=15, 
              cmap='Greens',
              extent= np.array([-1, 1, -1, 1]) * hor_lim_dva, #np.array([-hor_lim_dva,hor_lim_dva,-vert_lim_dva,vert_lim_dva]),#
              bins='log',
              linewidths=0.0625,
              edgecolors='black',
              alpha=1)

    ss.hexbin(np.hstack(right_xx_4plot), 
              np.hstack(right_yy_4plot),
              gridsize=15, 
              cmap='Reds', #'YlOrRd_r',
              extent= np.array([-1, 1, -1, 1]) * hor_lim_dva, #np.array([-hor_lim_dva,hor_lim_dva,-vert_lim_dva,vert_lim_dva]),#
              bins='log',
              linewidths=0.0625,
              edgecolors='black',
              alpha=0.5)
    
    ss.set_title('Visual field coverage for %s'%roi,fontsize=16)
    ss.set_xlabel('Horizontal visual position [dva]',fontsize=14)
    ss.set_ylabel('Vertical visual position [dva]',fontsize=14)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.tight_layout()
    #plt.ylim(-vert_lim_dva, vert_lim_dva) 
    # set middle lines
    ss.axvline(0, -hor_lim_dva, hor_lim_dva, lw=0.25, color='w')
    ss.axhline(0, -hor_lim_dva, hor_lim_dva, lw=0.25, color='w')
    
    # custom lines only to make labels
    custom_lines = [Line2D([0], [0], color='g',alpha=0.5, lw=4),
                    Line2D([0], [0], color='r',alpha=0.5, lw=4)]

    plt.legend(custom_lines, ['LH', 'RH'])

    plt.savefig(os.path.join(figure_out,'VF_coverage_ROI-%s_hemi-both.svg'%roi),dpi=100)
    
    f, ss = plt.subplots(1, 1, figsize=(8, 8), sharey=True)
    plt.hist(left_pa,color='g',alpha=0.5,label='LH')
    plt.hist(right_pa,color='r',alpha=0.5,label='RH')
    plt.xlabel('Polar angle')
    plt.legend()
    plt.title('Histogram of polar angle distribution for %s'%roi)
    #ss.axes.set_xlim(-np.pi,np.pi)

    plt.savefig(os.path.join(figure_out,'PA_histogram_ROI-%s_hemi-both.svg'%roi),dpi=100)
    
    # Visualise with polar histogram
    left_ind4plot = np.where((np.logical_not(np.isnan(left_pa))))
    right_ind4plot = np.where((np.logical_not(np.isnan(right_pa))))

    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection='polar'),figsize=(8, 8))
    rose_plot(ax, left_pa[left_ind4plot],color='g', lab_unit="radians")
    rose_plot(ax, right_pa[right_ind4plot],color='r', lab_unit="radians")
    plt.title('Histogram of polar angle distribution for %s'%roi, pad=20)

    fig.tight_layout()
    plt.savefig(os.path.join(figure_out,'PA_histogram_circular_ROI-%s_hemi-both.svg'%roi),dpi=100)
 

####### Check ECC distributions ###########

## ECD curves all pairs ##

# max eccentricity within screen dva
max_ecc = np.sqrt(vert_lim_dva**2 + hor_lim_dva**2)

# make list of roi pairs, to compare distributions, no repetitions
ks_roi_list = np.array([pair for pair in itertools.combinations(ROIs,2)])

for _,rois_ks in enumerate(ks_roi_list):
    
    # plot Empirical distribution functions
    #rois_ks = np.array(['V1','V2']) # rois to compare ecc distribution
    
    ecc1_4plot = []
    ecc2_4plot = []
    
    for w in range(len(estimates['subs'])): # loop once if one subject, or for all subjects when sj 'median'
        
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


## ECD for all rois in one plot ##

fig, ax = plt.subplots(1, 1, figsize=(16, 8), sharey=True)

for _,rois_ks in enumerate(ROIs):
    
    ecc_4plot = []
    
    for w in range(len(estimates['subs'])): # loop once if one subject, or for all subjects when sj 'all'
        
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
    
    for w in range(len(estimates['subs'])): # loop once if one subject, or for all subjects when sj 'all'
        
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


## Make distribuition difference matrix to compare all ROIs ###

# compute KS statistic for each ROI of each participant (in case of sj 'median')
# and append in dict

KS_stats = {}

for _,rois_ks in enumerate(ROIs): # compare ROI (ex V1)
    
    
    roi_stats = []
    
    for _,cmp_roi in enumerate(ROIs): # to all other ROIS
        
        sub_roi_stats = []
    
        for w in range(len(estimates['subs'])): # loop once if one subject, or for all subjects when sj 'all'

            new_rsq1 = masked_rsq[w][roi_verts[rois_ks]]
            new_ecc1 = masked_eccentricity[w][roi_verts[rois_ks]]
            indices4plot_1 = np.where((new_rsq1>rsq_threshold) & (np.logical_not(np.isnan(new_ecc1))))

            ecc_roi1 = np.sort(new_ecc1[indices4plot_1])

            new_rsq2 = masked_rsq[w][roi_verts[cmp_roi]]
            new_ecc2 = masked_eccentricity[w][roi_verts[cmp_roi]]
            indices4plot_2 = np.where((new_rsq2>rsq_threshold) & (np.logical_not(np.isnan(new_ecc2))))

            ecc_roi2 = np.sort(new_ecc2[indices4plot_2])

            sub_roi_stats.append(scipy.stats.ks_2samp(ecc_roi1, ecc_roi2)[0])
        
        roi_stats.append(np.median(sub_roi_stats)) # median max distance 
        
    KS_stats[rois_ks] = roi_stats
    
#Create DataFrame
DF_var = pd.DataFrame.from_dict(KS_stats).T
DF_var.columns = ROIs

# mask out repetitive values, making triangular matrix (still has identity diag)
for i,region in enumerate(ROIs):
    if i>0:
        for k in range(i):
            DF_var[region][k]=np.nan

# plot representational similarity matrix
fig, ax = plt.subplots(1, 1, figsize=(16, 8), sharey=True)

matrix = ax.matshow(DF_var)
plt.xticks(range(DF_var.shape[1]), DF_var.columns, fontsize=14)#, rotation=45)
plt.yticks(range(DF_var.shape[1]), DF_var.columns, fontsize=14)
fig.colorbar(matrix)
matrix.set_clim(vmin=0.1,vmax=0.4)

plt.title('Eccentricity distribution difference', fontsize=16, pad=20);
# This is very hack-ish
plt.gca().set_xticks([x - 0.5 for x in plt.gca().get_xticks()][1:], minor='true')
plt.gca().set_yticks([y - 0.5 for y in plt.gca().get_yticks()][1:], minor='true')
plt.grid(which='minor')

fig.savefig(os.path.join(figure_out,'RSA_ROI-all.svg'),dpi=100)


####### Make ECC vs size plot ###########

# create empty dataframe to store all relevant values for rois
all_roi = [] #pd.DataFrame(columns=['mean_ecc','mean_ecc_std','mean_size','mean_size_std','roi'])
n_bins = 10
min_ecc = 0.25
max_ecc = 3.3

for w in range(len(estimates['subs'])): # loop once if one subject, or for all subjects when sj 'median'
    
    for idx,roi in enumerate(ROIs):

        # get datapoints for RF only belonging to roi
        new_size = masked_size[w][roi_verts[roi]]
        new_ecc = masked_eccentricity[w][roi_verts[roi]]

        new_rsq = masked_rsq[w][roi_verts[roi]]

        # define indices of voxels within region to plot
        # with rsq > 0.17, and where value not nan, ecc values between 0.25 and 3.3
        indices4plot = np.where((new_ecc >= min_ecc) & (new_ecc<= max_ecc) & (new_rsq>rsq_threshold) & (np.logical_not(np.isnan(new_size))))[0]

        df = pd.DataFrame({'ecc': new_ecc[indices4plot],'size':new_size[indices4plot],
                               'rsq':new_rsq[indices4plot],'sub':np.tile(w,len(indices4plot))})
        
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

        if idx== 0 and w==0:
            all_roi = pd.DataFrame({'mean_ecc': mean_ecc,'mean_ecc_std':mean_ecc_std,
                                    'mean_size':mean_size,'mean_size_std':mean_size_std,
                                    'roi':np.tile(roi,n_bins),'sub':np.tile(w,n_bins)})
        else:
            all_roi = all_roi.append(pd.DataFrame({'mean_ecc': mean_ecc,'mean_ecc_std':mean_ecc_std,
                                                   'mean_size':mean_size,'mean_size_std':mean_size_std,
                                                   'roi':np.tile(roi,n_bins),'sub':np.tile(w,n_bins)}),ignore_index=True)

# get median bins for plotting (useful for correct median sub plot)
med_subs_df = []

for idx,roi in enumerate(ROIs):
      
    for j in range(n_bins):
        med_ecc = []
        med_ecc_std = []
        med_size = []
        med_size_std = []

        for w in range(len(estimates['subs'])):
            
            med_ecc.append(all_roi.loc[(all_roi['sub'] == w)& (all_roi['roi'] == roi)]['mean_ecc'].iloc[j])
            med_ecc_std.append(all_roi.loc[(all_roi['sub'] == w)& (all_roi['roi'] == roi)]['mean_ecc_std'].iloc[j])
            med_size.append(all_roi.loc[(all_roi['sub'] == w)& (all_roi['roi'] == roi)]['mean_size'].iloc[j])
            med_size_std.append(all_roi.loc[(all_roi['sub'] == w)& (all_roi['roi'] == roi)]['mean_size_std'].iloc[j])

        if idx== 0 and j==0:
            med_subs_df = pd.DataFrame({'med_ecc': [np.nanmedian(med_ecc)],'med_ecc_std':[np.nanmedian(med_ecc_std)],
                                    'med_size':[np.nanmedian(med_size)],'med_size_std':[np.nanmedian(med_size_std)],
                                    'roi':[roi]})
        else:
            med_subs_df = med_subs_df.append(pd.DataFrame({'med_ecc': [np.nanmedian(med_ecc)],'med_ecc_std':[np.nanmedian(med_ecc_std)],
                                                   'med_size':[np.nanmedian(med_size)],'med_size_std':[np.nanmedian(med_size_std)],
                                                   'roi':[roi]}),ignore_index=True)

fig, ax = plt.subplots(1, 1, figsize=(16, 8), sharey=True)

ax.axes.set_xlim(0,)
ax.axes.set_ylim(0,)

with sns.color_palette("husl", 11):
    ax = sns.lmplot(x='med_ecc', y='med_size', hue='roi',data=med_subs_df,scatter=False,height=8, aspect=1)
    
ax = plt.gca()
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
ax.set(xlabel='pRF eccentricity [dva]', ylabel='pRF size [dva]')
ax.set_title('ecc vs size plot, %d bins from %.2f-%.2f ecc [dva]'%(n_bins,min_ecc,max_ecc),fontsize=16)

fig1 = plt.gcf()
fig1.savefig(os.path.join(figure_out,'ecc_vs_size_binned_rsq-%0.2f.svg'%rsq_threshold), dpi=100,bbox_inches = 'tight')
    

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
    out_dir = os.path.join(figure_out,'sing_vox_fit')

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

    #if not os.path.exists(dm_filename):  # if not exists
    if sj in ['02','11','12','13']: # subjects that did pRF task with linux computer, so res was full HD
        screenshot2DM(png_filename, 0.1,
                  analysis_params['screenRes_HD'], dm_filename,dm_shape = 'square')  # create it

    else:
        screenshot2DM(png_filename, 0.1,
                    analysis_params['screenRes'], dm_filename,dm_shape = 'square')  # create it
    print('computed %s' % (dm_filename))

    #else:
    #    print('loading %s' % dm_filename)

    prf_dm = np.load(dm_filename)
    prf_dm = prf_dm.T # then it'll be (x, y, t)
    
    # change DM to see if fit is better like that
    # do new one which is average of every 2 TRs

    prf_dm = shift_DM(prf_dm)

    prf_dm = prf_dm[:,:,analysis_params['crop_pRF_TR']:] # crop DM because functional data also cropped now

    # define model params
    TR = analysis_params["TR"]

    hrf = utilities.spm_hrf(0,TR)

    # make stimulus object, which takes an input design matrix and sets up its real-world dimensions
    prf_stim = PRFStimulus2D(screen_size_cm=analysis_params["screen_width"], 
                             screen_distance_cm=analysis_params["screen_distance"], 
                             design_matrix=prf_dm, 
                             TR=TR)

    # sets up stimulus and hrf for this gridder
    if fit_model=='gauss':
        gg = Iso2DGaussianGridder(stimulus=prf_stim,
                                  hrf=hrf,
                                  filter_predictions=True,
                                  window_length=analysis_params["sg_filt_window_length"],
                                  polyorder=analysis_params["sg_filt_polyorder"],
                                  highpass=True,
                                  task_lengths=np.array([prf_dm.shape[-1]]))
    else:
        # css gridder
        gg = CSS_Iso2DGaussianGridder(stimulus=prf_stim,
                                          hrf=hrf,
                                          filter_predictions=True,
                                          window_length=analysis_params["sg_filt_window_length"],
                                          polyorder=analysis_params["sg_filt_polyorder"],
                                          highpass=True,
                                          task_lengths=np.array([prf_dm.shape[-1]]))

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
            new_rsq = masked_rsq[0][roi_verts[roi]]

            new_xx = masked_xx[0][roi_verts[roi]]
            new_yy = masked_yy[0][roi_verts[roi]]
            new_size = masked_size[0][roi_verts[roi]]

            new_beta = masked_beta[0][roi_verts[roi]]
            new_baseline = masked_baseline[0][roi_verts[roi]]

            new_ecc = masked_eccentricity[0][roi_verts[roi]]
            new_polar_angle = masked_polar_angle[0][roi_verts[roi]]
            
            new_ns = masked_ns[0][roi_verts[roi]]


            new_data = data[roi_verts[roi]] # data from ROI

            new_index = np.where(new_rsq==np.nanmax(new_rsq))[0][0]# index for max rsq within ROI

            timeseries = new_data[new_index]
            
            if fit_model=='gauss':
                model_it_prfpy = gg.return_single_prediction(new_xx[new_index],new_yy[new_index],new_size[new_index], 
                                                         new_beta[new_index],new_baseline[new_index])
            else: #css
                model_it_prfpy = gg.return_single_prediction(new_xx[new_index],new_yy[new_index],new_size[new_index], 
                                                         new_beta[new_index],new_baseline[new_index],new_ns[new_index])
                
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
            if fit_model=='gauss':
                plt.title('voxel %d of ROI %s , rsq of fit is %.3f' %(new_index,roi,new_rsq[new_index]))
            else:
                plt.title('voxel %d of ROI %s , rsq of fit is %.3f, n=%.2f' %(new_index,roi,new_rsq[new_index],new_ns[new_index]))
                
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
            new_rsq = masked_rsq[0][roi_verts[roi]]

            new_xx = masked_xx[0][roi_verts[roi]]
            new_yy = masked_yy[0][roi_verts[roi]]
            new_size = masked_size[0][roi_verts[roi]]

            new_beta = masked_beta[0][roi_verts[roi]]
            new_baseline = masked_baseline[0][roi_verts[roi]]

            new_ecc = masked_eccentricity[0][roi_verts[roi]]
            new_polar_angle = masked_polar_angle[0][roi_verts[roi]]
            
            new_ns = masked_ns[0][roi_verts[roi]]

            new_data = data[roi_verts[roi]] # data from ROI

            new_index =np.where(new_rsq==np.nanmax(new_rsq))[0][0]# index for max rsq within ROI

            timeseries = new_data[new_index]

            if fit_model=='gauss':
                model_it_prfpy = gg.return_single_prediction(new_xx[new_index],new_yy[new_index],new_size[new_index], 
                                                         new_beta[new_index],new_baseline[new_index])
            else: #css
                model_it_prfpy = gg.return_single_prediction(new_xx[new_index],new_yy[new_index],new_size[new_index], 
                                                         new_beta[new_index],new_baseline[new_index],new_ns[new_index])
                
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
                    axis.set_ylim(-3,6)#9)
                ax1 = axis
            else:
                if sj=='11':
                    axis.set_ylim(-1.5,3)
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
                                 subject='fsaverage_meridians', alpha=alpha)
#cortex.quickshow(images['polar'],with_curvature=True,with_sulci=True,with_colorbar=False)
filename = os.path.join(figure_out,'flatmap_space-fsaverage_rsq-%0.2f_type-polar_angle.svg' %rsq_threshold)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['polar'], recache=True,with_colorbar=False,with_curvature=True,with_sulci=True)

images['polar_noalpha'] = cortex.VertexRGB(rgb[..., 0].T, 
                                 rgb[..., 1].T, 
                                 rgb[..., 2].T, 
                                 subject='fsaverage_meridians', alpha=alpha_ones)
#cortex.quickshow(images['polar_noalpha'],with_curvature=True,with_sulci=True,with_colorbar=False)
filename = os.path.join(figure_out,'flatmap_space-fsaverage_rsq-%0.2f_type-polar_angle_noalpha.svg' %rsq_threshold)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['polar_noalpha'], recache=True,with_colorbar=False,with_curvature=True,with_sulci=True)


# vertex for rsq
images['rsq'] = cortex.Vertex2D(masked_rsq.T, alpha_ones, 'fsaverage_meridians',
                           vmin=0, vmax=1.0,
                           vmin2=0, vmax2=1.0, cmap='Reds_cov')
#cortex.quickshow(images['rsq'],with_curvature=True,with_sulci=True)
filename = os.path.join(figure_out,'flatmap_space-fsaverage_rsq-%0.2f_type-rsquared.svg' %rsq_threshold)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['rsq'], recache=True,with_colorbar=True,with_curvature=True,with_sulci=True)


# vertex for ecc
ecc4plot = masked_eccentricity.copy()
ecc4plot[alpha_mask] = np.nan

images['ecc'] = cortex.Vertex(ecc4plot.T, 'fsaverage_meridians',
                           vmin=0, vmax=6,
                           cmap='J4')
#cortex.quickshow(images['ecc'],with_curvature=True,with_sulci=True)
# Save this flatmap
filename = os.path.join(figure_out,'flatmap_space-fsaverage_rsq-%0.2f_type-eccentricity.svg' %rsq_threshold)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['ecc'], recache=True,with_colorbar=True,with_curvature=True,with_sulci=True)


# vertex for size
size4plot = masked_size.copy()
size4plot[alpha_mask] = np.nan

images['size'] = cortex.dataset.Vertex(size4plot.T, 'fsaverage_meridians',
                           vmin=0, vmax=6, #analysis_params['max_size'],
                           cmap='J4')
#cortex.quickshow(images['size'],with_curvature=True,with_sulci=True)
filename = os.path.join(figure_out,'flatmap_space-fsaverage_rsq-%0.2f_type-size.svg' %rsq_threshold)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['size'], recache=True,with_colorbar=True,with_curvature=True,with_sulci=True)






