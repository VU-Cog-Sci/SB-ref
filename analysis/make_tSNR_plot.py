
# quick script to make tSNR plots

import os, json
import sys, glob
import re 

import matplotlib.colors as colors

from utils import *

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as matcm
import matplotlib.pyplot as plt
from distutils.util import strtobool

from nilearn import surface

# define participant number and open json parameter file
if len(sys.argv)<2:	
    raise NameError('Please add subject number (ex:01) '
                    'as 1st argument in the command line!')
                    
elif len(sys.argv)<3:
    raise NameError('Please select task to compute tSNR plots ' 
                    'as 2nd argument in the command line!')
                    
elif len(sys.argv)<4:
    raise NameError('Please select if tSNR plot for median run (median) ' 
                    'or single runs (single) as 3rd argument in the command line!')
                             
else:	
    sj = str(sys.argv[1]).zfill(2) #fill subject number with 0 in case user forgets
    
    task = str(sys.argv[2])

    with open('analysis_params.json','r') as json_file:	
            analysis_params = json.load(json_file)	
            
    with_smooth = 'False'
    
    run_type = str(sys.argv[3])


# path to save plots
out_dir = os.path.join('/home/shared/2018/visual/SB-prep/SB-ref/derivatives/tSNR','sub-{sj}'.format(sj=sj),task)

if not os.path.exists(out_dir):  # check if path exists
    os.makedirs(out_dir)


# define paths and list of files to make plot of

# path to functional files
orig_filepath = glob.glob(os.path.join(analysis_params['fmriprep_dir'],'sub-{sj}'.format(sj=sj),'*','func/*'))
orig_filename = [run for run in orig_filepath if 'task-'+task in run and 'fsaverage' in run and run.endswith('.func.gii')]
orig_filename.sort()

post_filepath = glob.glob(os.path.join(analysis_params['post_fmriprep_outdir'], task, 'sub-{sj}'.format(sj=sj), '*'))
if with_smooth == 'True':
    file_extension = 'sg_smooth%d.func.gii'%analysis_params['smooth_fwhm']
else:
# last part of filename to use
    file_extension = '_sg.func.gii'

post_filename = [run for run in post_filepath if task in run and 'fsaverage' in run and run.endswith(file_extension)]
post_filename.sort()


# do same plots for pre and post processed files
for files in ['pre','post']:
    
    filename = orig_filename.copy() if files=='pre' else post_filename.copy() # choose correct list with absolute filenames
    
    gii_files = []
    if run_type == 'single':
        for run in range(6):
            gii_files.append([r for r in filename if 'run-'+str(run).zfill(2) in r])
            #print(gii_files)

    elif run_type == 'median':
    
        for field in ['hemi-L', 'hemi-R']: #choose one hemi at a time
            hemi = [h for h in filename if field in h and 'run-median' not in h] # make median run in output dir, 
                                                                        # but we don't want to average median run if already in original dir
            # set name for median run (now numpy array)
            med_file = os.path.join(out_dir, re.sub(
                'run-\d{2}_', 'run-median_', os.path.split(hemi[0])[-1]))
            # if file doesn't exist
            if not os.path.exists(med_file):
                gii_files.append(median_gii(hemi, out_dir))  # create it
                print('computed %s' % (gii_files))
            else:
                gii_files.append(med_file)
                print('median file %s already exists, skipping' % (gii_files))
        gii_files = [gii_files] # then format identical

    # load and combine both hemispheres
    for indx,list_pos in enumerate(gii_files):
        data_array = []
        if not list_pos:
            print('no files for run-%s'%str(indx).zfill(2))
        else:
            for val in list_pos :
                data_array.append(np.array(surface.load_surf_data(val))) #save both hemisphere estimates in same array
            data_array = np.vstack(data_array)
            
            new_filename = os.path.split(val)[-1].replace('hemi-R','hemi-both')
            print('making tSNR flatmap for %s'%str(new_filename))
            # make tsnr map
            stat_map=np.mean(data_array,axis=1)/np.std(data_array,axis=1)
            
            # save histogram of values
            fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')
            plt.hist(stat_map)
            plt.title('Histogram of tSNR values for %s run-%s of sub-%s'%(task,str(indx).zfill(2),sj))
            fig.savefig(os.path.join(out_dir,('histogram_'+new_filename).replace('.func.gii','.png')), dpi=100)
            
            up_lim = 200
            low_lim = 0
            colormap = 'viridis'

            # and plot it
            tsnr_flat = cortex.dataset.Vertex(stat_map.T, 'fsaverage',
                                 vmin=low_lim, vmax=up_lim, cmap=colormap)
            
            _ = cortex.quickflat.make_png(os.path.join(out_dir,('flatmap_'+new_filename).replace('.func.gii','.png')),
                                          tsnr_flat, recache=True,with_colorbar=True,with_curvature=True,with_sulci=True)

    
    


