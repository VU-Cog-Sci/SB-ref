
# make some relevant plots for soma
# essentially to check DM and plot timeseries + linear regression fit
# for single voxels
# and make interesting flatmaps from contrast 

import os, json
import sys, glob
import re 

import numpy as np
import pandas as pd

from utils import * #import script to use relevante functions

import cortex

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as matcm
import matplotlib.pyplot as plt

from statsmodels.stats import weightstats
from nistats.design_matrix import make_first_level_design_matrix
from nistats.contrasts import compute_contrast


# define participant number and open json parameter file
if len(sys.argv)<2:	
    raise NameError('Please add subject number (ex:01) '	
                    'as 1st argument in the command line!')	

else:	
    sj = str(sys.argv[1]).zfill(2) #fill subject number with 0 in case user forgets	

    with open('analysis_params.json','r') as json_file:	
            analysis_params = json.load(json_file)	


# define paths and variables
rsq_threshold = 0 #0.5 
z_threshold = analysis_params['z_threshold']

TR = analysis_params["TR"]

with_smooth = 'True'#analysis_params['with_smooth']

# last part of filename to use
file_extension = 'sg_psc.func.gii'

# make list with subject number (or all subject number if we want median contrast)
if sj == 'median':
    all_subs = ['01','02','03','04','05','08','09','11','12','13']
    all_contrasts = {}
else:
    all_subs = [sj]


# dir to get contrasts
contrast_dir = os.path.join(analysis_params['soma_outdir'],'new_fits','sub-{sj}'.format(sj=sj))
# path to save figures
figure_out = os.path.join(analysis_params['derivatives'],'figures','soma','new_fits','sub-{sj}'.format(sj=sj))

if not os.path.exists(figure_out): # check if path to save figures exists
    os.makedirs(figure_out) 


# plot single voxel for each broad region (for single subject)
if sj != 'median': 
    
    # load contrasts for different regions
    face_contrast = np.load(os.path.join(contrast_dir,'z_face_contrast_rsq-%.2f.npy' %rsq_threshold))
    hand_contrast = np.load(os.path.join(contrast_dir,'z_upper_limb_contrast_rsq-%.2f.npy' %rsq_threshold))
    leg_contrast = np.load(os.path.join(contrast_dir,'z_lower_limb_contrast_rsq-%.2f.npy' %rsq_threshold))

    
    # find max z-score
    face_index =  np.where(face_contrast == max(face_contrast))[0][0] 
    if sj=='11': face_index = 325606 # because it's a nice one to plot, to use in figure (rsq = 93% so also good fit)
    hand_index = np.where(hand_contrast == max(hand_contrast))[0][0] 
    leg_index = np.where(leg_contrast == max(leg_contrast))[0][0] 

    # path to events
    eventdir = os.path.join(analysis_params['sourcedata_dir'],'sub-{sj}'.format(sj=str(sj).zfill(2)),'ses-01','func')
    print('event files from %s' % eventdir)
    
    # make average event file
    events_avg = make_median_soma_events(sj,eventdir)
    
    # path to functional files
    filepath = os.path.join(analysis_params['post_fmriprep_outdir'], 'soma', 'sub-{sj}'.format(sj=str(sj).zfill(2)))
    
    # load data array for sub    
    data = make_median_soma_sub(sj,filepath,contrast_dir,file_extension=file_extension)
    
    # load estimates array (to get model timecourse)
    estimates = np.load([os.path.join(contrast_dir,x) for x in os.listdir(contrast_dir) if 'estimates.npz' in x][0],allow_pickle=True)
    print('loading estimates array from %s'%contrast_dir)
    
    # plot single voxel timecourses
    time_sec = np.linspace(0,data.shape[-1]*TR,num=data.shape[-1]) # array with 141 timepoints, in seconds

    task = ['face','upper_limb','lower_limb']
    vertex = [face_index,hand_index,leg_index]
    colors = ['#db3050','#0093b7','#9066ba']

    for i in range(len(vertex)):

        # legend labels for data
        if task[i]=='face':
            dlabel = 'face' 
        elif task[i]=='upper_limb':
            dlabel = 'hand'
        else:
            dlabel = 'leg'

        # plot data with model
        fig= plt.figure(figsize=(15,7.5),dpi=100)
        plt.plot(time_sec,estimates['model'][vertex[i]],c='#0093b7',lw=3,label='GLM',zorder=1)
        plt.scatter(time_sec,data[vertex[i]], marker='.',c='k',label=dlabel)
        plt.xlabel('Time (s)',fontsize=18)
        plt.ylabel('BOLD signal change (%)',fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlim(0,len(data[vertex[i]])*TR)
        plt.title('voxel %d (%s) , MSE = %.3f, rsq = %.3f' %(vertex[i],task[i],estimates['mse'][vertex[i]],estimates['r2'][vertex[i]]))
        plt.legend(loc=0)


        if dlabel!='leg': # issue for leg timing, check later
            # plot axis vertical bar on background to indicate stimulus display time
            stim_onset = []
            for w in range(len(events_avg)):
                if events_avg['trial_type'][w] in analysis_params['all_contrasts'][task[i]]:
                    stim_onset.append(events_avg['onset'][w])
            stim_onset = list(set(stim_onset)); stim_onset.sort()  # to remove duplicate values (both hands situation)

            ax_count = 0
            for h in range(6):
                incr = 3 if task[i]=='face' else 4 # increment for vertical bar (to plot it from index 0 to index 4)
                plt.axvspan(stim_onset[ax_count], stim_onset[ax_count+3]+2.25, facecolor='b', alpha=0.1)
                ax_count += 4 if task[i]=='face' else 5


        fig.savefig(os.path.join(figure_out,'soma_singvoxfit_timeseries_%s.svg'%task[i]), dpi=100,bbox_inches = 'tight')
        
    # now combine tasks in both (to have option of different plots)
    task = ['face','upper_limb']
    vertex = [face_index,hand_index] 
    blue_color = ['#004759','#00a7d1']
    data_color = ['#262626','#8a8a8a']

    fig, axis = plt.subplots(1,figsize=(15,7.5),dpi=100)

    for i in range(len(task)):

        # legend labels for data
        dlabel = 'face' if task[i]=='face' else 'hand'

        # instantiate a second axes that shares the same x-axis
        #if task[i]=='upper_limb': axis = axis.twinx() 

        # plot data with model
        axis.plot(time_sec,estimates['model'][vertex[i]],c=blue_color[i],lw=3,label=dlabel+', R$^2$=%.2f'%estimates['r2'][vertex[i]],zorder=1)
        axis.scatter(time_sec,data[vertex[i]], marker='v',s=15,c=blue_color[i])#,label=dlabel)
        axis.set_xlabel('Time (s)',fontsize=18)
        axis.set_ylabel('BOLD signal change (%)',fontsize=18)
        axis.tick_params(axis='both', labelsize=14)
        #axis.tick_params(axis='y', labelcolor=blue_color[i])
        axis.set_xlim(0,len(data[vertex[i]])*TR)

        #if task[i]=='upper_limb':
        #    axis.set_ylim(-2,3.5) 
        #else:
        #    axis.set_ylim(-4,7)

        # plot axis vertical bar on background to indicate stimulus display time

        stim_onset = []
        for w in range(len(events_avg)):
            if events_avg['trial_type'][w] in analysis_params['all_contrasts'][task[i]]:
                stim_onset.append(events_avg['onset'][w])
        stim_onset = list(set(stim_onset)); stim_onset.sort()  # to remove duplicate values (both hands situation)

        if i == 0:
            handles,labels = axis.axes.get_legend_handles_labels()
        else:
            a,b = axis.axes.get_legend_handles_labels()
            handles = handles+a
            labels = labels+b

        ax_count = 0
        for h in range(6):
            incr = 3 if task[i]=='face' else 4 # increment for vertical bar (to plot it from index 0 to index 4)
            plt.axvspan(stim_onset[ax_count], stim_onset[ax_count+3]+2.25, facecolor=blue_color[i], alpha=0.1)
            ax_count += 4 if task[i]=='face' else 5


    axis.legend(handles,labels,loc='upper left')  # doing this to guarantee that legend is how I want it   

    fig.savefig(os.path.join(figure_out,'soma_singvoxfit_timeseries_%s.svg'%str(task)), dpi=100,bbox_inches = 'tight')

# load rsq values
rsq_filename = os.path.join(contrast_dir,'rsq.npy')
if not os.path.isfile(rsq_filename):
    rsq_all = []
    for _,sub in enumerate (all_subs): # for sub or all subs

        file_dir = os.path.join(os.path.split(contrast_dir)[0],'sub-'+sub)

        # load estimates array (to get model timecourse)
        estimates = np.load([os.path.join(file_dir,x) for x in os.listdir(file_dir) if 'estimates.npz' in x][0],allow_pickle=True)
        print('loading estimates array from %s'%file_dir)

        # seems stupid, but necessary to take ou the dtype object info (pycortex doesnt work otherwise)
        rsq = estimates['r2']; rsq = np.vstack(rsq); rsq=rsq[...,0] 
        rsq_all.append(rsq)

    rsq = np.median(np.array(rsq_all),axis=0)
    np.save(rsq_filename,rsq)
else:
    print('loading rsq from %s'%rsq_filename)
    rsq = np.load(rsq_filename,allow_pickle=True)


# for other plots we might want to use smoothed data
if with_smooth=='True':
    contrast_dir = os.path.join(contrast_dir,'smooth%d'%analysis_params['smooth_fwhm'])
    figure_out = os.path.join(figure_out,'smooth%d'%analysis_params['smooth_fwhm'])
    
    if not os.path.exists(figure_out): # check if path to save figures exists
        os.makedirs(figure_out) 

    # smooth rsq 
    if not os.path.isfile(os.path.join(contrast_dir,'rsq.npy')):
        rsq = smooth_nparray(os.path.join(analysis_params['post_fmriprep_outdir'], 'soma', 'sub-{sj}'.format(sj=str(all_subs[0]).zfill(2))),
                           rsq,
                           contrast_dir,
                           'rsq',
                           'rsq.npy',
                            n_TR=141,
                           file_extension=file_extension,
                           sub_mesh='fsaverage',
                           smooth_fwhm=analysis_params['smooth_fwhm'])
    else:
        print('load %s'%os.path.join(contrast_dir,'rsq.npy'))
        rsq = np.load(os.path.join(contrast_dir,'rsq.npy'))


# now make some flatmaps

alpha_mask = np.array([True if val<= rsq_threshold or np.isnan(val) else False for _,val in enumerate(rsq)])

# plot rsq values in cortex
alpha_ones = np.ones(alpha_mask.shape) #np.clip(rsq / np.nanmax(rsq) * 3, 0, 1) #* 3, 0, 1) #np.ones(alpha_mask.shape)
alpha_ones[alpha_mask] = np.nan

# vertex for rsq
images = {}

images['rsq'] = cortex.Vertex2D(rsq, alpha_ones, 'fsaverage_gross',
                           vmin=0, vmax=1.0,
                           vmin2=0, vmax2=1.0, cmap='Reds_cov')

filename = os.path.join(figure_out,'flatmap_space-fsaverage_rsq-%0.2f_type-rsquared.svg' %rsq_threshold)
#cortex.quickshow(images['rsq'],with_curvature=True,with_sulci=True)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['rsq'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True)


# load contrasts for different regions
face_contrast = np.load(os.path.join(contrast_dir,'z_face_contrast_rsq-%.2f.npy' %(rsq_threshold)))
hand_contrast = np.load(os.path.join(contrast_dir,'z_upper_limb_contrast_rsq-%.2f.npy' %(rsq_threshold)))
leg_contrast = np.load(os.path.join(contrast_dir,'z_lower_limb_contrast_rsq-%.2f.npy' %(rsq_threshold)))

# plot different body areas
# but first threshold them (only show positive z-scores)
data_threshed_face = zthresh(face_contrast,threshold=z_threshold,side='above')
data_threshed_hand = zthresh(hand_contrast,threshold=z_threshold,side='above')
data_threshed_leg = zthresh(leg_contrast,threshold=z_threshold,side='above')


# vertex for face vs all others
images['v_face'] = cortex.Vertex(face_contrast, 'fsaverage_gross',
                           vmin=0, vmax=7,
                           cmap='Blues')

#cortex.quickshow(images['v_face'],with_curvature=True,with_sulci=True,with_colorbar=True)
filename = os.path.join(figure_out,'flatmap_space-fsaverage_rsq-%0.2f_type-face-vs-all.svg' %(rsq_threshold))
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_face'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True)

# vertex for face vs all others - with zscore thresholding
images['v_face'] = cortex.Vertex(data_threshed_face, 'fsaverage_gross',
                           vmin=0, vmax=7,
                           cmap='Blues')

#cortex.quickshow(images['v_face'],with_curvature=True,with_sulci=True,with_colorbar=True)
filename = os.path.join(figure_out,'flatmap_space-fsaverage_rsq-%0.2f_zscore-%.2f_type-face-vs-all.svg' %(rsq_threshold,z_threshold))
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_face'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True)


# vertex for hand vs all others
images['v_hand'] = cortex.Vertex(hand_contrast, 'fsaverage_gross',
                           vmin=0, vmax=7,
                           cmap='Blues')

#cortex.quickshow(images['v_hand'],with_curvature=True,with_sulci=True,with_colorbar=True)
filename = os.path.join(figure_out,'flatmap_space-fsaverage_rsq-%0.2f_type-hands-vs-all.svg' %(rsq_threshold))
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_hand'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True)

# vertex for hand vs all others - with zscore thresholding
images['v_hand'] = cortex.Vertex(data_threshed_hand, 'fsaverage',
                           vmin=0, vmax=7,
                           cmap='Blues')

#cortex.quickshow(images['v_hand'],with_curvature=True,with_sulci=True,with_colorbar=True)
filename = os.path.join(figure_out,'flatmap_space-fsaverage_rsq-%0.2f_zscore-%.2f_type-hands-vs-all.svg' %(rsq_threshold,z_threshold))
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_hand'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True)


# vertex for leg vs all others
images['v_leg'] = cortex.Vertex(data_threshed_leg, 'fsaverage_gross',
                           vmin=0, vmax=7,
                           cmap='Blues')

#cortex.quickshow(images['v_leg'],with_curvature=True,with_sulci=True,with_colorbar=True)
filename = os.path.join(figure_out,'flatmap_space-fsaverage_rsq-%0.2f_type-legs-vs-all.svg' %(rsq_threshold))
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_leg'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True)


# vertex for leg vs all others - with zscore thresholding
images['v_leg'] = cortex.Vertex(leg_contrast, 'fsaverage_gross',
                           vmin=0, vmax=7,
                           cmap='Blues')

#cortex.quickshow(images['v_leg'],with_curvature=True,with_sulci=True,with_colorbar=True)
filename = os.path.join(figure_out,'flatmap_space-fsaverage_rsq-%0.2f_zscore-%.2f_type-legs-vs-all.svg' %(rsq_threshold,z_threshold))
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_leg'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True)


# combine 3 body part maps, threshold values
combined_zvals = np.array((data_threshed_face,data_threshed_hand,data_threshed_leg))

print('Computing center of mass for different regions combined')
soma_labels, soma_zval = zsc_2_COM(combined_zvals)

images['v_combined_alpha'] = cortex.Vertex(soma_labels, 'fsaverage_gross',
                           vmin=0, vmax=2,
                           cmap='rainbow')#BROYG_2D')#'my_autumn')

#cortex.quickshow(images['v_combined_alpha'],with_curvature=True,with_sulci=True,with_colorbar=True)
filename = os.path.join(figure_out,'flatmap_space-fsaverage_rsq-%0.2f_zscore-%.2f_type-combined-face-hands-legs.svg' %(rsq_threshold,z_threshold))
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_combined_alpha'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True)


# load within region contrasts
## Right vs left
RLupper_zscore = np.load(os.path.join(contrast_dir,'z_right-left_hand_contrast_thresh-%0.2f_rsq-%.2f.npy'%(z_threshold,rsq_threshold)))
RLlower_zscore = np.load(os.path.join(contrast_dir,'z_right-left_leg_contrast_thresh-%0.2f_rsq-%.2f.npy'%(z_threshold,rsq_threshold)))

# make red blue costum colormap
# not sure if needed but ok for now
n_bins = 100  # Discretizes the interpolation into bins
RBalpha_dict = {'red':  ((0.0, 0.0, 0.0),
                   (0.25, 0.0, 0.0),
                   (0.5, 0.8, 1.0),
                   (0.75, 1.0, 1.0),
                   (1.0, 0.4, 1.0)),

         'green': ((0.0, 0.0, 0.0),
                   (0.25, 0.0, 0.0),
                   (0.5, 0.9, 0.9),
                   (0.75, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'blue':  ((0.0, 0.0, 0.4),
                   (0.25, 1.0, 1.0),
                   (0.5, 1.0, 0.8),
                   (0.75, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),
          
          'alpha': ((0.0, 1.0, 1.0),
                   (0.25,1.0, 1.0),
                    (0.5, 0.3, 0.3),
                   (0.75,1.0, 1.0),
                    (1.0, 1.0, 1.0))
        }

blue_red = LinearSegmentedColormap('BlueRed_alpha', RBalpha_dict, N=n_bins)
matcm.register_cmap(name='BlueRed_alpha', cmap=blue_red) # register it in matplotlib lib


# # vertex for right vs left hand
# images['rl_upper'] = cortex.Vertex2D(RLupper_zscore,alpha_ones, 'fsaverage_gross',
#                            vmin=-7, vmax=7,
#                            vmin2=0, vmax2=1,
#                            cmap='BuBkRd_alpha_2D')

images['rl_upper'] = cortex.Vertex(RLupper_zscore, 'fsaverage',
                          vmin=-7, vmax=7,
                          cmap=blue_red)

#cortex.quickshow(images['rl_upper'],with_curvature=True,with_sulci=True,with_colorbar=True)
filename = os.path.join(figure_out,'flatmap_space-fsaverage_rsq-%0.2f_zscore-%.2f_type-RL-hands.svg' %(rsq_threshold,z_threshold))
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['rl_upper'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True)


# # vertex for right vs left leg
# images['rl_lower'] = cortex.Vertex2D(RLlower_zscore,alpha_ones, 'fsaverage_gross',
#                            vmin=-7, vmax=7,
#                            vmin2=0, vmax2=1,
#                            cmap='BuBkRd_alpha_2D')

images['rl_lower'] = cortex.Vertex(RLlower_zscore, 'fsaverage',
                          vmin=-7, vmax=7,
                          cmap=blue_red)

#cortex.quickshow(images['rl_lower'],with_curvature=True,with_sulci=True,with_colorbar=True)
filename = os.path.join(figure_out,'flatmap_space-fsaverage_rsq-%0.2f_zscore-%.2f_type-RL-legs.svg' %(rsq_threshold,z_threshold))
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['rl_lower'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True)


# all fingers in hand combined
LHfing_zscore = [] # load and append each finger z score in left hand list
RHfing_zscore = [] # load and append each finger z score in right hand list


print('Loading data for all fingers and appending in list')

for i in range(len(analysis_params['all_contrasts']['upper_limb'])//2):
    
    LHfing_zscore.append(np.load(os.path.join(contrast_dir,'z_%s-all_lhand_contrast_thresh-%0.2f_rsq-%.2f.npy' 
                                 %(analysis_params['all_contrasts']['upper_limb'][i],z_threshold,rsq_threshold))))
    RHfing_zscore.append(np.load(os.path.join(contrast_dir,'z_%s-all_rhand_contrast_thresh-%0.2f_rsq-%.2f.npy' 
                                              %(analysis_params['all_contrasts']['upper_limb'][i+5],z_threshold,rsq_threshold))))
   


LHfing_zscore = np.array(LHfing_zscore)
RHfing_zscore = np.array(RHfing_zscore)

# compute center of mass and appropriate z-scores for each hand
print('Computing center of mass for left hand fingers')
LH_COM , LH_avgzval = zsc_2_COM(LHfing_zscore)
print('Computing center of mass for right hand fingers')
RH_COM , RH_avgzval = zsc_2_COM(RHfing_zscore)


# all fingers left hand combined ONLY in left hand region 
# (as defined by LvsR hand contrast values)
images['v_Lfingers'] = cortex.Vertex(LH_COM, 'fsaverage_gross',
                           vmin=0, vmax=4,
                           cmap='J4')#costum colormap added to database

#cortex.quickshow(images['v_Lfingers'],with_curvature=True,with_sulci=True,with_colorbar=True)
filename = os.path.join(figure_out,'flatmap_space-fsaverage_rsq-%0.2f_type-LH-fingers1-5.svg' %(rsq_threshold))
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_Lfingers'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True)


# all fingers right hand combined ONLY in right hand region 
# (as defined by LvsR hand contrast values)
images['v_Rfingers'] = cortex.Vertex(RH_COM, 'fsaverage_gross',
                           vmin=0, vmax=4,
                           cmap='J4')#costum colormap added to database

#cortex.quickshow(images['v_Rfingers'],with_curvature=True,with_sulci=True,with_colorbar=True)
filename = os.path.join(figure_out,'flatmap_space-fsaverage_rsq-%0.2f_type-RH-fingers1-5.svg' %(rsq_threshold))
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_Rfingers'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True)


# # threshold left vs right, to only show relevant vertex 
# # (i.e., where zscore is "significant", use it to mask hands for plotting)
# data_threshed_RLhand=zthresh(RLupper_zscore,threshold=z_threshold,side='both')

# RLhand_mask = np.array([True if np.isnan(val) else False for _,val in enumerate(data_threshed_RLhand)])

# LH_COM_4plot = LH_COM.copy()
# LH_COM_4plot[RLhand_mask] = np.nan

# # all fingers left hand combined ONLY in left hand region 
# # (as defined by LvsR hand contrast values)
# images['v_Lfingers'] = cortex.Vertex(LH_COM_4plot, 'fsaverage_gross',
#                            vmin=0, vmax=4,
#                            cmap='J4')#costum colormap added to database

# #cortex.quickshow(images['v_Lfingers'],with_curvature=True,with_sulci=True,with_colorbar=True)
# filename = os.path.join(figure_out,'flatmap_space-fsaverage_rsq-%0.2f_zscore-%.2f_type-LH-fingers1-5.svg' %(rsq_threshold,z_threshold))
# print('saving %s' %filename)
# _ = cortex.quickflat.make_png(filename, images['v_Lfingers'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True)


# RH_COM_4plot = RH_COM.copy()
# RH_COM_4plot[RLhand_mask] = np.nan

# # all fingers left hand combined ONLY in left hand region 
# # (as defined by LvsR hand contrast values)
# images['v_Rfingers'] = cortex.Vertex(RH_COM_4plot, 'fsaverage_gross',
#                            vmin=0, vmax=4,
#                            cmap='J4')#costum colormap added to database

# #cortex.quickshow(images['v_Lfingers'],with_curvature=True,with_sulci=True,with_colorbar=True)
# filename = os.path.join(figure_out,'flatmap_space-fsaverage_rsq-%0.2f_zscore-%.2f_type-RH-fingers1-5.svg' %(rsq_threshold,z_threshold))
# print('saving %s' %filename)
# _ = cortex.quickflat.make_png(filename, images['v_Rfingers'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True)


# all individual face regions combined

allface_zscore = [] # load and append each face part z score in list

print('Loading data for each face part and appending in list')

for _,name in enumerate(analysis_params['all_contrasts']['face']):
    
    facedata = np.load(os.path.join(contrast_dir,'z_%s-other_face_areas_contrast_thresh-%0.2f_rsq-%.2f.npy' %(name,z_threshold,rsq_threshold)))   
    allface_zscore.append(facedata)  

allface_zscore = np.array(allface_zscore)

# combine them all in same array

print('Computing center of mass for face elements %s' %(analysis_params['all_contrasts']['face']))
allface_COM , allface_avgzval = zsc_2_COM(allface_zscore)


# 'eyebrows', 'eyes', 'mouth','tongue', , combined
images['v_facecombined'] = cortex.Vertex(allface_COM, 'fsaverage_gross',
                           vmin=0, vmax=3,
                           cmap='J4') #costum colormap added to database

#cortex.quickshow(images['v_facecombined'],with_curvature=True,with_sulci=True,with_colorbar=True)
filename = os.path.join(figure_out,'flatmap_space-fsaverage_rsq-%0.2f_type-eyebrows-eyes-mouth-tongue.svg' %(rsq_threshold))
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_facecombined'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True)


# threshold left vs right, to only show relevant vertex 
# (i.e., where zscore is "significant", use it to mask hands for plotting)

face_mask = np.array([True if np.isnan(val) else False for _,val in enumerate(data_threshed_face)])

allface_COM_4plot = allface_COM.copy()
allface_COM_4plot[face_mask] = np.nan

# 'eyebrows', 'eyes', 'mouth','tongue', , combined
images['v_facecombined'] = cortex.Vertex(allface_COM_4plot, 'fsaverage_gross',
                           vmin=0, vmax=3,
                           cmap='J4') #costum colormap added to database

#cortex.quickshow(images['v_facecombined'],with_curvature=True,with_sulci=True,with_colorbar=True)
filename = os.path.join(figure_out,'flatmap_space-fsaverage_rsq-%0.2f_zscore-%.2f_type-eyebrows-eyes-mouth-tongue.svg' %(rsq_threshold,z_threshold))
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_facecombined'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True)


print('Done!')















