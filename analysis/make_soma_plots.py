
# make some relevant plots for soma
# essentially to check DM and plot timeseries + linear regression fit
# for single voxels

# initial script, should look better but works

import os, json
import sys, glob
import re 

import numpy as np
import pandas as pd

import nibabel as nb
from nilearn import surface

from nistats.design_matrix import make_first_level_design_matrix
from nistats.first_level_model import run_glm
from nistats.contrasts import compute_contrast


from utils import * #import script to use relevante functions

from nistats.reporting import plot_design_matrix
from scipy.stats import pearsonr


# define participant number and open json parameter file
if len(sys.argv)<2:	
    raise NameError('Please add subject number (ex:01) '	
                    'as 1st argument in the command line!')	

else:	
    sj = str(sys.argv[1]).zfill(2) #fill subject number with 0 in case user forgets	

    with open('analysis_params.json','r') as json_file:	
            analysis_params = json.load(json_file)	


# define paths
figure_out = os.path.join(analysis_params['derivatives'],'figures','soma','sub-{sj}'.format(sj=sj))

if not os.path.exists(figure_out): # check if path to save figures exists
    os.makedirs(figure_out) 

with_smooth = 'False'#analysis_params['with_smooth']


##### compute median run for soma and load data #####

# path to functional files
filepath = glob.glob(os.path.join(analysis_params['post_fmriprep_outdir'], 'soma', 'sub-{sj}'.format(sj=sj), '*'))
print('functional files from %s' % os.path.split(filepath[0])[0])

# last part of filename to use
file_extension = 'sg_psc.func.gii'

# list of functional files (5 runs)
filename = [run for run in filepath if 'soma' in run and 'fsaverage' in run and run.endswith(file_extension)]
filename.sort()

# path to save fits, for testing
out_dir = os.path.join(analysis_params['soma_outdir'],'sub-{sj}'.format(sj=sj),'tests')

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

##########

# load contrasts for different regions
post_soma_path = os.path.join(analysis_params['post_fmriprep_outdir'],'soma','sub-{sj}'.format(sj=str(sj).zfill(2)))
eventdir = os.path.join(analysis_params['sourcedata_dir'],'sub-{sj}'.format(sj=str(sj).zfill(2)),'ses-01','func')
median_path = os.path.join(analysis_params['soma_outdir'],'sub-{sj}'.format(sj=str(sj).zfill(2)),'run-median')

face_contrast = np.load(os.path.join(median_path,'z_face_contrast.npy'))
hand_contrast = np.load(os.path.join(median_path,'z_upper_limb_contrast.npy'))
leg_contrast = np.load(os.path.join(median_path,'z_lower_limb_contrast.npy'))


# find max z-score
min_zthresh = 8 #9 #because max one had too high values, maybe vein?
face_index =  np.where(face_contrast == max(face_contrast))[0][0] #np.where(face_contrast > min_zthresh)[0][0] 
hand_index = np.where(hand_contrast == max(hand_contrast))[0][0] #np.where(hand_contrast > min_zthresh)[0][0]
leg_index = np.where(leg_contrast == max(leg_contrast))[0][0] #np.where(leg_contrast > min_zthresh)[0][1] 


# stuff to plot
task = ['face','hand','leg']
vertex = [face_index,hand_index,leg_index]
colors = ['#db3050','#0093b7','#9066ba']

# plot the timecourses
plot_soma_timecourse(sj,'median',task,vertex,post_soma_path,eventdir,
                     figure_out,plotcolors = colors,template='fsaverage',extension=file_extension)


# make DM

# list of stimulus onsets
events = [os.path.join(eventdir,run) for run in os.listdir(eventdir) if 'soma' in run and run.endswith('events.tsv')]
events.sort()

TR = analysis_params["TR"]


# make function that makes median event data frame for x runs of sub or for median sub
# now this will do
onsets_allsubs = []
durations_allsubs = []

all_events = []
for _,val in enumerate(events):

    events_pd = pd.read_csv(val,sep = '\t')

    new_events = []

    for ev in events_pd.iterrows():
        row = ev[1]   
        if row['trial_type'][0] == 'b': # if both hand/leg then add right and left events with same timings
            new_events.append([row['onset'],row['duration'],'l'+row['trial_type'][1:]])
            new_events.append([row['onset'],row['duration'],'r'+row['trial_type'][1:]])
        else:
            new_events.append([row['onset'],row['duration'],row['trial_type']])

    df = pd.DataFrame(new_events, columns=['onset','duration','trial_type'])  #make sure only relevant columns present
    all_events.append(df)

# make median event dataframe
onsets = []
durations = []
for w in range(len(all_events)):
    onsets.append(all_events[w]['onset'])
    durations.append(all_events[w]['duration'])

onsets_allsubs.append(np.median(np.array(onsets),axis=0)) #append average onset of all runs
durations_allsubs.append(np.median(np.array(durations),axis=0))


# all subjects in one array, use this to compute contrasts
events_avg = pd.DataFrame({'onset':np.median(np.array(onsets_allsubs),axis=0),'duration':np.median(np.array(durations_allsubs),axis=0),'trial_type':all_events[0]['trial_type']})

# specifying the timing of fMRI frames
frame_times = TR * (np.arange(data.shape[-1]))

# Create the design matrix, hrf model containing Glover model 
design_matrix = make_first_level_design_matrix(frame_times,
                                               events=events_avg,
                                               hrf_model='glover'
                                               )

# plot design matrix
plot = plot_design_matrix(design_matrix)
fig = plot.get_figure()
fig.savefig(os.path.join(figure_out,'design_matrix.svg'), dpi=100,bbox_inches = 'tight')


time_sec = np.linspace(0,data.shape[-1]*TR,num=data.shape[-1]) # array with 90 timepoints, in seconds

# plot predictors for different regions

for _,t in enumerate(analysis_params['all_contrasts'].keys()):
    # get indices of relevant predictors from DM to plot on data
    predictor_ind = []
    for i,part in enumerate(design_matrix.keys()):
        if part in analysis_params['all_contrasts'][t]:
            predictor_ind.append(i)
    predictor_ind = np.array(predictor_ind)
    print(predictor_ind)
    fig= plt.figure(figsize=(15,7.5),dpi=100)
    for _,ind in enumerate(predictor_ind):
        plt.plot(time_sec,design_matrix.values[...,ind],label=design_matrix.keys()[ind])
    plt.legend(loc=0)
    plt.xlabel('Time (s)',fontsize=18)
    plt.title('Convolved Predictors for %s region'%t)
    fig.savefig(os.path.join(figure_out,'predictors_convolved_%s.svg'%t), dpi=100,bbox_inches = 'tight')


 # also plot all predictors in same, because why not?
fig= plt.figure(figsize=(15,7.5),dpi=100)
for ind in range(len(design_matrix.keys())):
        plt.plot(time_sec,design_matrix.values[...,ind],label=design_matrix.keys()[ind])
plt.legend(loc=0)
plt.xlabel('Time (s)',fontsize=18)
plt.title('Convolved Predictors')
fig.savefig(os.path.join(figure_out,'predictors_convolved_all.svg'), dpi=100,bbox_inches = 'tight')


# not sure of what run_glm does and how to access betas from the regression results
# so get regression results from numpy least squares linear regress function
task = ['face','upper_limb','lower_limb']

for i in range(len(vertex)):
    
    betas_conv = np.linalg.lstsq(design_matrix.values, data[vertex[i]])[0]
    model_sig = design_matrix.values.dot(betas_conv)
    
    mse = np.mean((model_sig - data[vertex[i]]) ** 2) # calculate mean of squared residuals
    r2 = pearsonr(model_sig, data[vertex[i]])[0] ** 2 # and the rsq

    # legend labels for data
    if task[i]=='face':
        dlabel = 'face' 
    elif task[i]=='upper_limb':
        dlabel = 'hand'
    else:
        dlabel = 'leg'
    
    # plot data with model
    fig= plt.figure(figsize=(15,7.5),dpi=100)
    plt.plot(time_sec,model_sig,c='#0093b7',lw=3,label='GLM',zorder=1)
    plt.scatter(time_sec,data[vertex[i]], marker='.',c='k',label=dlabel)
    plt.xlabel('Time (s)',fontsize=18)
    plt.ylabel('BOLD signal change (%)',fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(0,len(data[vertex[i]])*TR)
    plt.title('voxel %d (%s) , MSE = %.3f, rsq = %.3f' %(vertex[i],task[i],mse,r2))
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
fig, axis = plt.subplots(1,figsize=(15,7.5),dpi=100)
task = ['face','upper_limb']
blue_color = ['#004759','#00a7d1']
data_color = ['#262626','#8a8a8a']
for i in range(len(task)):
    
    betas_conv = np.linalg.lstsq(design_matrix.values, data[vertex[i]])[0]
    model_sig = design_matrix.values.dot(betas_conv)
    
    mse = np.mean((model_sig - data[vertex[i]]) ** 2) # calculate mean of squared residuals
    r2 = pearsonr(model_sig, data[vertex[i]])[0] ** 2 # and the rsq
    
    # legend labels for data
    dlabel = 'face' if task[i]=='face' else 'hand'
    
    # plot data with model
    axis.plot(time_sec,model_sig,c=blue_color[i],lw=3,label=dlabel,zorder=1)
    axis.scatter(time_sec,data[vertex[i]], marker='v',s=15,c=blue_color[i])#,label=dlabel)
    axis.set_xlabel('Time (s)',fontsize=18)
    axis.set_ylabel('BOLD signal change (%)',fontsize=18)
    axis.tick_params(axis='both', labelsize=14)
    axis.set_xlim(0,len(data[vertex[i]])*TR)
    #plt.title('voxel %d (%s) , MSE = %.3f, rsq = %.3f' %(vertex[i],task[i],mse,r2))
    
    # plot axis vertical bar on background to indicate stimulus display time
    stim_onset = []
    for w in range(len(events_avg)):
        if events_avg['trial_type'][w] in analysis_params['all_contrasts'][task[i]]:
            stim_onset.append(events_avg['onset'][w])
    stim_onset = list(set(stim_onset)); stim_onset.sort()  # to remove duplicate values (both hands situation)
            
    ax_count = 0
    for h in range(6):
        incr = 3 if task[i]=='face' else 4 # increment for vertical bar (to plot it from index 0 to index 4)
        plt.axvspan(stim_onset[ax_count], stim_onset[ax_count+3]+2.25, facecolor=blue_color[i], alpha=0.1)
        ax_count += 4 if task[i]=='face' else 5

handles,labels = axis.axes.get_legend_handles_labels()
axis.legend([handles[0],handles[1]],
            [labels[0],labels[1]],loc='upper left')  # doing this to guarantee that legend is how I want it   
fig.savefig(os.path.join(figure_out,'soma_singvoxfit_timeseries_%s.svg'%str(task)), dpi=100,bbox_inches = 'tight')










