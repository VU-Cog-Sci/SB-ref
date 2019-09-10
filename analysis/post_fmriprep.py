
# extra processing after fmriprep, for all tasks

import os, json
import sys, glob
import re 

import numpy as np
import pandas as pd

from utils import * #import script to use relevante functions

# define participant number and open json parameter file
if len(sys.argv)<2:	
    raise NameError('Please add subject number (ex:01) '	
                    'as 1st argument in the command line!')	
   	 
else:	
    sj = str(sys.argv[1]).zfill(2) #fill subject number with 0 in case user forgets	

    with open('analysis_params.json','r') as json_file:	
            analysis_params = json.load(json_file)	
    
# define paths and list of files
filepath = glob.glob(os.path.join(analysis_params['fmriprep_dir'],'sub-{sj}'.format(sj=sj),'*','func/*'))
tasks = ['fn','prf']#,'soma','rlb','rli','rs']

for t,cond in enumerate(tasks):

    # list of functional files
    filename = [run for run in filepath if 'task-'+tasks[t] in run and 'fsaverage' in run and run.endswith('.func.gii')]
    filename.sort()
    # list of confounds
    confounds = [run for run in filepath if 'task-'+tasks[t] in run and run.endswith('_desc-confounds_regressors.tsv')]
    confounds.sort()
    
    if not filename: # if list empty
        print('Subject %s has no files for %s' %(sj,cond))

    else:
    
        TR = analysis_params["TR"]

        # set output path for processed files
        outpath = os.path.join(analysis_params['post_fmriprep_outdir'],tasks[t],'sub-{sj}'.format(sj=sj))

        if not os.path.exists(outpath): # check if path to save median run exist
            os.makedirs(outpath) 
            
        # make loop for length of filenames

        for _,file in enumerate(filename):

            # define hemisphere to plot
            hemi='left' if '_hemi-L' in file else 'right'
            
            # plot all steps as sanity check
            plot_tSNR(file,hemi,os.path.join(outpath,'tSNR'),mesh='fsaverage')
            
            # high pass filter all runs (savgoy-golay)
            filt_gii,filt_gii_pth = highpass_gii(file,analysis_params['sg_filt_polyorder'],analysis_params['sg_filt_deriv'],
                                                         analysis_params['sg_filt_window_length'],outpath)

            plot_tSNR(filt_gii_pth,hemi,os.path.join(outpath,'tSNR'),mesh='fsaverage')
            
            if cond == 'prf' or 'fn': # don't clean confounds for prf or fn.. doenst help retino maps(?)
                clean_gii = filt_gii
                clean_gii_pth = filt_gii_pth
            else: #regress out PCA of confounds from data
                # first sg filter them
                filt_conf = highpass_confounds(confounds,analysis_params['nuisance_columns'],analysis_params['sg_filt_polyorder'],analysis_params['sg_filt_deriv'],
                                                       analysis_params['sg_filt_window_length'],TR,outpath)
                # NEED TO CHECK THIS ONE STILL, AND CHANGE FUNCTION TO MAKE SURE IT SAVES ONLY GII
                clean_gii, clean_gii_pth = clean_confounds(filt_gii_pth,filt_conf,outpath,combine_hemi=False) 
                
            # do PSC
            psc_data,psc_data_pth = psc_gii(clean_gii_pth,outpath, method='median') 

            plot_tSNR(psc_data_pth,hemi,os.path.join(outpath,'tSNR'),mesh='fsaverage')
            
            # smooth it
            smt_file, smt_pth = smooth_gii(psc_data_pth,outpath,fwhm=5)
            
            plot_tSNR(smt_pth,hemi,os.path.join(outpath,'tSNR'),mesh='fsaverage')









