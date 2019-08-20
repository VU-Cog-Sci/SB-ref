
# extra processing after fmriprep, for all tasks

import os, json
import sys, glob
import re 

import numpy as np
import pandas as pd

from utils import * #import script to use relevante functions

import nipype.interfaces.freesurfer as fs


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
tasks = ['fn','soma','prf','rlb','rli','rs']
# do PSC?      
with_psc = analysis_params['with_psc']

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

        # high pass filter all runs (savgoy-golay)
        filt_gii,filt_gii_pth = highpass_gii(filename,analysis_params['sg_filt_polyorder'],analysis_params['sg_filt_deriv'],
                                             analysis_params['sg_filt_window_length'],outpath,combine_hemi=False)

        if cond == 'prf' or 'fn': # don't clean confounds for prf or fn.. doenst help retino maps(?)
        	clean_gii = filt_gii
        	clean_gii_pth = filt_gii_pth
        else: #regress out PCA of confounds from data
            # first sg filter them
            filt_conf = highpass_confounds(confounds,analysis_params['nuisance_columns'],analysis_params['sg_filt_polyorder'],analysis_params['sg_filt_deriv'],
                                           analysis_params['sg_filt_window_length'],TR,outpath)
        	clean_gii, clean_gii_pth = clean_confounds(filt_gii_pth,filt_conf,outpath,combine_hemi=False) 

        if with_psc=='True':
            final_gii = psc_gii(clean_gii_pth,outpath, method='median') 
        else:
            final_gii = clean_gii_pth
            
        # transform to mgz to smooth it
        mgz_files = nparray2mgz(final_gii,filename,outpath)

        # now smooth it
        mgz_files.sort()

        for index,mgz in enumerate(mgz_files):
            smoother = fs.SurfaceSmooth()
            smoother.inputs.in_file = mgz
            smoother.inputs.subject_id = 'fsaverage'
            if index % 2 == 0: #even indexs are left hemi
                smoother.inputs.hemi = 'lh'
            else:
                smoother.inputs.hemi = 'rh'
            smoother.inputs.fwhm = 5
            smoother.run() # doctest: +SKIP
    
            new_filename = os.path.splitext(os.path.split(mgz)[-1])[0]+'_smooth%d.mgz'%(smoother.inputs.fwhm)
            os.rename(os.path.join(os.getcwd(),new_filename), os.path.join(outpath,new_filename)) #move to correct dir








