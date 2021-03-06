
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


# define paths and relevant variables

with_smooth = 'False'#analysis_params['with_smooth']
rsq_threshold = 0.5 
z_threshold = analysis_params['z_threshold']


# changes depending on data used
if with_smooth=='True':
    # dir to get contrasts
    contrast_dir = os.path.join(analysis_params['soma_outdir'],'new_fits','sub-{sj}'.format(sj=sj),'smooth%d'%analysis_params['smooth_fwhm'])
    # path to save figures
    figure_out = os.path.join(analysis_params['derivatives'],'figures','soma','new_fits','sub-{sj}'.format(sj=sj),'smooth%d'%analysis_params['smooth_fwhm'])
    # last part of filename to use
    file_extension = 'sg_psc_smooth%d.func.gii'%analysis_params['smooth_fwhm']
else:
    # dir to get contrasts
    contrast_dir = os.path.join(analysis_params['soma_outdir'],'new_fits','sub-{sj}'.format(sj=sj))
    # path to save figures
    figure_out = os.path.join(analysis_params['derivatives'],'figures','soma','new_fits','sub-{sj}'.format(sj=sj))
    # last part of filename to use
    file_extension = 'sg_psc.func.gii'

if not os.path.exists(figure_out): # check if path to save figures exists
    os.makedirs(figure_out) 


# load contrasts for different regions

face_contrast = np.load(os.path.join(contrast_dir,'z_face_contrast.npy'))
hand_contrast = np.load(os.path.join(contrast_dir,'z_upper_limb_contrast.npy'))
leg_contrast = np.load(os.path.join(contrast_dir,'z_lower_limb_contrast.npy'))

# find max z-score
face_index =  np.where(face_contrast == max(face_contrast))[0][0] 
if sj=='11': face_index = 325606 # because it's a nice one to plot, to use in figure (rsq = 93% so also good fit)
hand_index = np.where(hand_contrast == max(hand_contrast))[0][0] 
leg_index = np.where(leg_contrast == max(leg_contrast))[0][0] 


if sj == 'median':

    all_subs = ['01','02','03','04','05','08','09','11','12','13']

else:
    all_subs = [sj]

# load data array for sub    
data = make_median_soma_sub(all_subs,file_extension,contrast_dir,median_gii=median_gii)

# load events dataframe for sub 
events_avg = make_median_soma_events(all_subs)

# load estimates array (to get model timecourse)
estimates = np.load([os.path.join(contrast_dir,x) for x in os.listdir(contrast_dir) if 'estimates.npz' in x][0],allow_pickle=True)
print('loading estimates array from %s'%contrast_dir)


# plot single voxel timecourses

TR = analysis_params["TR"]
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


# now make some flatmaps

alpha_mask = np.array([True if val<= rsq_threshold or np.isnan(val) else False for _,val in enumerate(estimates['r2'])])

# plot rsq values in cortex
alpha_ones = np.ones(alpha_mask.shape)
alpha_ones[alpha_mask] = np.nan

# seems stupid, but necessary to take ou the dtype object info (pycortex doesnt work otherwise)
rsq = estimates['r2']; rsq = np.vstack(rsq); rsq=rsq[...,0] 

# vertex for rsq
images = {}

images['rsq'] = cortex.Vertex2D(rsq, alpha_ones, 'fsaverage',
                           vmin=0, vmax=1.0,
                           vmin2=0, vmax2=1.0, cmap='Reds_cov')
#images['rsq'] = cortex.Vertex(rsq,'fsaverage',
#                           vmin=0, vmax=1.0,
#                           cmap='Reds')
#cortex.quickshow(images['rsq'],with_curvature=True,with_sulci=True)
filename = os.path.join(figure_out,'flatmap_space-fsaverage_rsq-%0.2f_type-rsquared.svg' %rsq_threshold)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['rsq'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True)

# plot different body areas
# but first threshold them (only show positive z-scores)
data_threshed_face = zthresh(face_contrast,threshold=z_threshold,side='above')
data_threshed_hand = zthresh(hand_contrast,threshold=z_threshold,side='above')
data_threshed_leg = zthresh(leg_contrast,threshold=z_threshold,side='above')

# vertex for face vs all others
images['v_face'] = cortex.Vertex(data_threshed_face.T, 'fsaverage',
                           vmin=0, vmax=7,
                           cmap='Blues')

#cortex.quickshow(images['v_face'],with_curvature=True,with_sulci=True,with_colorbar=True)
filename = os.path.join(figure_out,'flatmap_space-fsaverage_rsq-%0.2f_zscore-%.2f_type-face-vs-all.svg' %(rsq_threshold,z_threshold))
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_face'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True)

# vertex for face vs all others
images['v_hand'] = cortex.Vertex(data_threshed_hand.T, 'fsaverage',
                           vmin=0, vmax=7,
                           cmap='Blues')

#cortex.quickshow(images['v_hand'],with_curvature=True,with_sulci=True,with_colorbar=True)
filename = os.path.join(figure_out,'flatmap_space-fsaverage_rsq-%0.2f_zscore-%.2f_type-hands-vs-all.svg' %(rsq_threshold,z_threshold))
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_hand'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True)

# vertex for face vs all others
images['v_leg'] = cortex.Vertex(data_threshed_leg.T, 'fsaverage',
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

images['v_combined_alpha'] = cortex.Vertex(soma_labels, 'fsaverage',
                           vmin=0, vmax=2,
                           cmap='rainbow')#BROYG_2D')#'my_autumn')


#images['v_combined_alpha'] = cortex.Vertex2D(soma_labels, soma_zval, 'fsaverage',
#                           vmin=0, vmax=2,
#                           vmin2=min(soma_zval), vmax2=7, cmap='autumn_alpha')#BROYG_2D')#'my_autumn')

#cortex.quickshow(images['v_combined_alpha'],with_curvature=True,with_sulci=True,with_colorbar=True)
filename = os.path.join(figure_out,'flatmap_space-fsaverage_rsq-%0.2f_zscore-%.2f_type-combined-face-hands-legs.svg' %(rsq_threshold,z_threshold))
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_combined_alpha'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True)

## Right vs left
RLupper_zscore = np.load(os.path.join(contrast_dir,'z_right-left_hand_contrast_thresh-%0.2f.npy'%z_threshold))
RLlower_zscore = np.load(os.path.join(contrast_dir,'z_right-left_leg_contrast_thresh-%0.2f.npy'%z_threshold))

# threshold left vs right, to only show relevant vertex
data_threshed_RLhand=zthresh(RLupper_zscore,side='both')
data_threshed_RLleg=zthresh(RLlower_zscore,side='both')

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

# vertex for right vs left hand
images['rl_upper'] = cortex.Vertex(data_threshed_RLhand.T, 'fsaverage',
                           vmin=-7, vmax=7,
                           cmap=blue_red)

#cortex.quickshow(images['rl_upper'],with_curvature=True,with_sulci=True,with_colorbar=True)
filename = os.path.join(figure_out,'flatmap_space-fsaverage_rsq-%0.2f_zscore-%.2f_type-RL-hands.svg' %(rsq_threshold,z_threshold))
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['rl_upper'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True)

# vertex for right vs left leg
images['rl_lower'] = cortex.Vertex(data_threshed_RLleg.T, 'fsaverage',
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
    
    LHfing_zscore.append(np.load(os.path.join(contrast_dir,'z_%s-all_lhand_contrast_thresh-%0.2f.npy' 
                                 %(analysis_params['all_contrasts']['upper_limb'][i],z_threshold))))
    RHfing_zscore.append(np.load(os.path.join(contrast_dir,'z_%s-all_rhand_contrast_thresh-%0.2f.npy' 
                                              %(analysis_params['all_contrasts']['upper_limb'][i+5],z_threshold))))
   


LHfing_zscore = np.array(LHfing_zscore)
RHfing_zscore = np.array(RHfing_zscore)

# compute center of mass and appropriate z-scores for each hand
print('Computing center of mass for left hand fingers')
LH_COM , LH_avgzval = zsc_2_COM(LHfing_zscore)
print('Computing center of mass for right hand fingers')
RH_COM , RH_avgzval = zsc_2_COM(RHfing_zscore)


# all fingers left hand combined ONLY in left hand region 
# (as defined by LvsR hand contrast values)
images['v_Lfingers'] = cortex.Vertex(LH_COM, 'fsaverage',
                           vmin=0, vmax=4,
                           cmap='J4')#costum colormap added to database

#cortex.quickshow(images['v_Lfingers'],with_curvature=True,with_sulci=True,with_colorbar=True)
filename = os.path.join(figure_out,'flatmap_space-fsaverage_rsq-%0.2f_zscore-%.2f_type-LH-fingers1-5.svg' %(rsq_threshold,z_threshold))
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_Lfingers'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True)


# all fingers right hand combined ONLY in right hand region 
# (as defined by LvsR hand contrast values)
images['v_Rfingers'] = cortex.Vertex(RH_COM, 'fsaverage',
                           vmin=0, vmax=4,
                           cmap='J4')#costum colormap added to database

#cortex.quickshow(images['v_Rfingers'],with_curvature=True,with_sulci=True,with_colorbar=True)
filename = os.path.join(figure_out,'flatmap_space-fsaverage_rsq-%0.2f_zscore-%.2f_type-RH-fingers1-5.svg' %(rsq_threshold,z_threshold))
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_Rfingers'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True)


# all individual face regions combined

allface_zscore = [] # load and append each face part z score in list

print('Loading data for each face part and appending in list')

for _,name in enumerate(analysis_params['all_contrasts']['face']):
    
    facedata = np.load(os.path.join(contrast_dir,'z_%s-other_face_areas_contrast_thresh-%0.2f.npy' %(name,z_threshold)))   
    allface_zscore.append(facedata)  

allface_zscore = np.array(allface_zscore)

# combine them all in same array

print('Computing center of mass for face elements %s' %(analysis_params['all_contrasts']['face']))
allface_COM , allface_avgzval = zsc_2_COM(allface_zscore)


# 'eyebrows', 'eyes', 'mouth','tongue', , combined
images['v_facecombined'] = cortex.Vertex(allface_COM.T, 'fsaverage',
                           vmin=0, vmax=3,
                           cmap='J4') #costum colormap added to database

#cortex.quickshow(images['v_facecombined'],with_curvature=True,with_sulci=True,with_colorbar=True)
filename = os.path.join(figure_out,'flatmap_space-fsaverage_rsq-%0.2f_zscore-%.2f_type-eyebrows-eyes-mouth-tongue.svg' %(rsq_threshold,z_threshold))
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_facecombined'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True)


# make beta weights for different ROIs

# get vertices for subject fsaverage
# for later plotting

ROIs = ['Medial','Frontal','Parietal']#,'Insula'] # list will combine those areas, string only accounts for 1 ROI

roi_verts = {} #empty dictionary 
for i,val in enumerate(ROIs):
    roi_verts[val] = cortex.get_roi_verts('fsaverage_gross',val)[val]

# get betas and rsq values for later
betas = estimates['betas']
rsq = estimates['r2']

# get DM predictor keys

# specifying the timing of fMRI frames
frame_times = TR * (np.arange(data.shape[-1]))
# Create the design matrix, hrf model containing Glover model 
design_matrix = make_first_level_design_matrix(frame_times,
                                               events=events_avg,
                                               hrf_model='glover'
                                               )

regressors =np.array(design_matrix.keys()).astype(str)
# make and save average beta weights for different ROIs
for idx,roi in enumerate(ROIs):
    # plot beta weights for single voxel
    fig, axis = plt.subplots(1,figsize=(25,7.5),dpi=100)
    
    # get the average values 
    # weight voxels based on rsq of fit
    beta_avg = weightstats.DescrStatsW(betas[roi_verts[roi]],weights=rsq[roi_verts[roi]]).mean

    # need to do this to get weighted standard deviations of the mean for each regressor
    beta_std = []
    for w in range(len(regressors)):
        beta_pred =  np.array([betas[roi_verts[roi]][x][w] for x in range(len(betas[roi_verts[roi]]))])
        beta_std.append(weightstats.DescrStatsW(beta_pred,weights=rsq[roi_verts[roi]]).std_mean) 
                         

    y_pos = np.arange(len(regressors))
    plt.bar(y_pos,beta_avg,yerr=np.array(beta_std), align='center')
    plt.xticks(y_pos, regressors)
    
    axis.set_title('Averaged beta weights for each regressor in ROI %s'%(roi))
    #plt.show()
    fig.savefig(os.path.join(figure_out,'average_betas_all_regressors_ROI-%s.svg'%roi), dpi=100,bbox_inches = 'tight')


# now combine predictors of same region in bar plots 

reg_keys = list(analysis_params['all_contrasts'].keys()); reg_keys.sort() # list of key names (of different body regions)
for idx,roi in enumerate(ROIs):
  fig, axis = plt.subplots(1,figsize=(15,7.5),dpi=100)
  region_beta_avg = []
  region_beta_std = []
  for idx,region in enumerate(reg_keys):#reg_keys):
      region_betas = []
      region_rsq = []
      for f,regr in enumerate(regressors): # join all betas for region in same array (and rsq for weights)
          if regr in analysis_params['all_contrasts'][region]:
              region_betas.append(np.array([betas[roi_verts[roi]][x][f] for x in range(len(betas[roi_verts[roi]]))]))
              region_rsq.append(np.array(rsq[roi_verts[roi]]))
              
      region_beta_avg.append(weightstats.DescrStatsW(np.array(region_betas).ravel(),weights=np.array(region_rsq).ravel()).mean)
      region_beta_std.append(weightstats.DescrStatsW(np.array(region_betas).ravel(),weights=np.array(region_rsq).ravel()).std_mean)
      
      
  y_pos = np.arange(len(reg_keys))
  plt.bar(y_pos,region_beta_avg,yerr=np.array(region_beta_std), align='center')
  plt.xticks(y_pos, reg_keys)

  axis.set_title('Averaged beta weights for regressors of regions combined in ROI %s'%(roi))
  fig.savefig(os.path.join(figure_out,'average_betas_combined_regressors_ROI-%s.svg'%roi), dpi=100,bbox_inches = 'tight')
  #plt.show()


# repetitive, fix once defined which plots should stay from these beta bar plots ######

reg_keys = list(analysis_params['all_contrasts'].keys()) # list of key names (of different body regions)
for idx,roi in enumerate(ROIs):
    fig, axis = plt.subplots(1,figsize=(15,7.5),dpi=100)
  
    region_beta_avg = []
    region_beta_std = []
    labels = []
    for idx,region in enumerate(reg_keys):#reg_keys):
        #region_betas = []
        #region_rsq = []
        for f,regr in enumerate(regressors): # join all betas for region in same array (and rsq for weights)
    
            if regr in analysis_params['all_contrasts'][region]:
                print('computing avg betas for %s'%regr)
                region_betas = np.array([betas[roi_verts[roi]][x][f] for x in range(len(betas[roi_verts[roi]]))])
                region_rsq = np.array(rsq[roi_verts[roi]])
                
                region_beta_avg.append(weightstats.DescrStatsW(np.array(region_betas).ravel(),weights=np.array(region_rsq).ravel()).mean)
                region_beta_std.append(weightstats.DescrStatsW(np.array(region_betas).ravel(),weights=np.array(region_rsq).ravel()).std_mean)
                
                labels.append(regr)

    left_beta_avg = [x for ind,x in enumerate(region_beta_avg) if ('rhand' not in labels[ind]) if ('rleg' not in labels[ind])]
    left_beta_std = [x for ind,x in enumerate(region_beta_std) if ('rhand' not in labels[ind]) if ('rleg' not in labels[ind])]
    left_labels = [x for ind,x in enumerate(labels) if ('rhand' not in labels[ind]) if ('rleg' not in labels[ind])]

    right_beta_avg = [x for ind,x in enumerate(region_beta_avg) if ('lhand' not in labels[ind]) if ('lleg' not in labels[ind])]
    right_beta_std = [x for ind,x in enumerate(region_beta_std) if ('lhand' not in labels[ind]) if ('lleg' not in labels[ind])]
    right_labels = [x for ind,x in enumerate(labels) if ('lhand' not in labels[ind]) if ('lleg' not in labels[ind])]

    # width of the bars
    barWidth = 0.3

    # The x position of bars
    y_pos = np.arange(len(left_labels))
    y_pos2 = [x + barWidth for x in y_pos]

    plt.bar(y_pos,left_beta_avg,yerr=np.array(left_beta_std), align='center',color = 'yellow', edgecolor = 'black',width = barWidth,label='left')
    plt.bar(y_pos2,right_beta_avg,yerr=np.array(right_beta_std), align='center',color = 'blue', edgecolor = 'black',width = barWidth,label='right')

    plt.xticks([r + barWidth for r in y_pos], ['eyebrows','eyeblinks','mouth','tongue','fing1','fing2','fing3','fing4','fing5','leg'])#left_labels)
    axis.set_ylim(-0.5,1.25)
    #plt.xticks(y_pos, left_labels)
    axis.set_title('Averaged beta weights for relevant regressors in ROI %s'%(roi))
    plt.legend()
    #plt.show()

    fig.savefig(os.path.join(figure_out,'average_betas_ordered_regressors_ROI-%s.svg'%roi), dpi=100,bbox_inches = 'tight')

print('Done!')















