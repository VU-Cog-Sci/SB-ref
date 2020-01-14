
# just fit one voxel from subject
# to see differences in performance from both procedures


import os

import json
import sys
import glob
import re

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy as np
import scipy as sp
import scipy.stats as stats
import nibabel as nb
from nilearn.image import mean_img

from nilearn import surface

from utils import *  # import script to use relevante functions

# requires pfpy be installed - preferably with python setup.py develop
from prfpy.rf import *
from prfpy.timecourse import *
from prfpy.stimulus import PRFStimulus2D
from prfpy.grid import Iso2DGaussianGridder
from prfpy.fit import Iso2DGaussianFitter

import matplotlib.gridspec as gridspec
import scipy

## need to add below to get hrf function from there
from popeye import utilities 

# define participant number and open json parameter file
if len(sys.argv)<2:	
    raise NameError('Please add subject number (ex:01) '	
                    'as 1st argument in the command line!')	

elif len(sys.argv)<3:	
    raise NameError('Please add hemi (left vs right) '	
                    'as 2nd argument in the command line!')	

elif len(sys.argv)<4:	
    raise NameError('Please voxel index number '	
                	'as 3rd argument in the command line!')	

else:	
    sj = str(sys.argv[1]).zfill(2) #fill subject number with 0 in case user forgets	

    hemifield = str(sys.argv[2]) #hemifield

    index = int(sys.argv[3])	

    with open('analysis_params.json','r') as json_file:	
            analysis_params = json.load(json_file)	


# path to functional files
filepath = glob.glob(os.path.join(analysis_params['post_fmriprep_outdir'], 'prf', 'sub-{sj}'.format(sj=sj), '*'))
print('functional files from %s' % os.path.split(filepath[0])[0])

# path to save fits, for testing
out_dir = os.path.join(analysis_params['pRF_outdir'],'sub-{sj}'.format(sj=sj),'tests')

if not os.path.exists(out_dir):  # check if path exists
    os.makedirs(out_dir)


with_smooth = 'False'

# last part of filename to use
file_extension = 'cropped_sg_psc.func.gii'

# list of functional files (5 runs)
filename = [run for run in filepath if 'prf' in run and 'fsaverage' in run and run.endswith(file_extension)]
filename.sort()

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
if hemifield == 'left':
    hemi = 'hemi-L'
elif hemifield == 'right':
    hemi = 'hemi-R'

gii_file = med_gii[0] if hemi == 'hemi-L' else  med_gii[1]
print('using %s' %gii_file)

data = np.array(surface.load_surf_data(gii_file))

# plot one voxel from that hemi, will be the one used for fitting
timeseries = data[index,...]
print('voxel %d belonging to %s' %(index,hemi))

fig= plt.figure(figsize=(15,7.5))

plt.plot(range(len(timeseries)),timeseries, linestyle='--', marker='o',c='k')
plt.xlabel('Time (TR)',fontsize=18)
plt.ylabel('BOLD signal change (%)',fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(0,len(timeseries))
plt.title('Voxel %d from %s'%(index,hemi))
fig.savefig(os.path.join(out_dir,'timeseries_voxel-%d_%s.png'%(index,hemi)), dpi=100,bbox_inches = 'tight')

# do this to have same shape of data when used in fitting scripts
timeseries = timeseries[np.newaxis,:]
timeseries.shape

# create/load design matrix for prfpy
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

# shift DM to be the average of every 2 TRs
prf_dm = shift_DM(prf_dm)
prf_dm = prf_dm[:,:,analysis_params['crop_pRF_TR']:] # crop DM because functional data also cropped now

np.save(os.path.join(out_dir,'prf_dm_square_shift.npy'), prf_dm)

# define model params
TR = analysis_params["TR"]

hrf = utilities.spm_hrf(0,TR)

# make stimulus object, which takes an input design matrix and sets up its real-world dimensions
prf_stim = PRFStimulus2D(screen_size_cm=analysis_params["screen_width"], 
                         screen_distance_cm=analysis_params["screen_distance"], 
                         design_matrix=prf_dm, 
                         TR=TR)

# sets up stimulus and hrf for this gridder
gg = Iso2DGaussianGridder(stimulus=prf_stim,
                          hrf=hrf,
                          filter_predictions=False,
                          window_length=analysis_params["sg_filt_window_length"],
                          polyorder=analysis_params["sg_filt_polyorder"],
                          highpass=False)

# set grid parameters
grid_nr = analysis_params["grid_steps"]
sizes = analysis_params["max_size"] * np.linspace(np.sqrt(analysis_params["min_size"]/analysis_params["max_size"]),1,grid_nr)**2
eccs = analysis_params["max_eccen"] * np.linspace(np.sqrt(analysis_params["min_eccen"]/analysis_params["max_eccen"]),1,grid_nr)**2
polars = np.linspace(0, 2*np.pi, grid_nr)
n_list = [0.25,0.5,0.75,1]

gf = Iso2DGaussianFitter(data=timeseries, gridder=gg, n_jobs=1, fit_css=True)

#filename for the numpy array with the estimates of the grid fit
grid_estimates_filename = gii_file.replace('.func.gii', '_voxel%d_estimates_cropped_shifted.npz'%index)

#if not os.path.isfile(grid_estimates_filename): # if estimates file doesn't exist
print('%s not found, fitting grid'%grid_estimates_filename)

# gaussian fit
# do grid fit and save estimates
gf.grid_fit(ecc_grid=eccs,
            polar_grid=polars,
            size_grid=sizes,
            n_grid=n_list)

np.savez(grid_estimates_filename,
         x = gf.gridsearch_params[..., 0],
         y = gf.gridsearch_params[..., 1],
         size = gf.gridsearch_params[..., 2],
         betas = gf.gridsearch_params[...,3],
         baseline = gf.gridsearch_params[..., 4],
         ns = gf.gridsearch_params[..., 5],
         r2 = gf.gridsearch_params[..., 6])

# plot model
# load estimates for each case
grid_estimates = np.load(grid_estimates_filename)

# grid estimates for that index

x_grid = grid_estimates['x'][0]
y_grid = grid_estimates['y'][0]
sigma_grid = grid_estimates['size'][0]
baseline_grid = grid_estimates['baseline'][0]
beta_grid = grid_estimates['betas'][0]
rsq_grid = grid_estimates['r2'][0]

model_grid = gg.return_single_prediction(x_grid,y_grid,sigma_grid,beta=beta_grid,baseline=baseline_grid)
model_grid = model_grid[0]

# plot data with model
fig= plt.figure(figsize=(15,7.5))
plt.plot(model_grid,c='b',lw=3,label='model grid',zorder=1)
plt.plot(timeseries[0],c='k',linestyle='--', marker='o',label='data')#,zorder=2)
plt.xlabel('Time (TR)',fontsize=18)
plt.ylabel('BOLD signal change (%)',fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(0,len(timeseries[0]))
plt.title('GRID FIT for Voxel %d, %s, RSQ %0.2f for prfpy'%(index,hemi,rsq_grid))
plt.legend(loc=0)
fig.savefig(os.path.join(out_dir,'timeseries_voxel-%d_%s_gridfit_shifted.png'%(index,hemi)), dpi=100,bbox_inches = 'tight')

# CSS iterative fit
#gf_css = gf
#gf_css.fit_css=True

print('doing iterative fit')
gf.iterative_fit(rsq_threshold=0.1, verbose=False,
                 gridsearch_params = gf.gridsearch_params)

iterative_CSS_out = gii_file.replace('.func.gii', '_voxel%d_iterative_CSS_output_cropped_shifted.npz'%index)
np.savez(iterative_CSS_out,
         x = gf.iterative_search_params[..., 0],
         y = gf.iterative_search_params[..., 1],
         size = gf.iterative_search_params[..., 2],
         betas = gf.iterative_search_params[...,3],
         baseline = gf.iterative_search_params[..., 4],
         ns = gf.iterative_search_params[..., 5],
         r2 = gf.iterative_search_params[..., 6])

# Now compare iterative fits
it_estimates_css = np.load(iterative_CSS_out)

# iterative estimates for that index
x_it_css = it_estimates_css['x'][0]
y_it_css = it_estimates_css['y'][0]
sigma_it_css = it_estimates_css['size'][0]
baseline_it_css = it_estimates_css['baseline'][0]
beta_it_css = it_estimates_css['betas'][0]
rsq_it_css = it_estimates_css['r2'][0]


# plot both models
model_it_css = gg.return_single_prediction(x_it_css ,y_it_css ,sigma_it_css ,beta=beta_it_css,baseline=baseline_it_css)
model_it_css = model_it_css[0]

# plot data with model
fig= plt.figure(figsize=(15,7.5))
plt.plot(model_it_css ,c='b',lw=3,label='model prfpy',zorder=1)
plt.plot(timeseries[0],c='k',linestyle='--', marker='o',label='data')#,zorder=2)
plt.xlabel('Time (TR)',fontsize=18)
plt.ylabel('BOLD signal change (%)',fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(0,len(timeseries[0]))
plt.title('Iterative FIT for Voxel %d, %s, RSQ %0.2f for prfpy'%(index,hemi,rsq_it_css))# and %0.2f for popeye'%(index,hemi,rsq_it_prfpy,rsq_it_pop))
plt.legend(loc=0)

fig.savefig(os.path.join(out_dir,'timeseries_voxel-%d_%s_iterativefit_CSS_shifted.png'%(index,hemi)), dpi=100,bbox_inches = 'tight')

# add this so then I can see which bar passes correspond to model peaks
# hence check if it makes sense
# plot RF for voxel and bar passes corresponding to model peaks
sig_peaks = scipy.signal.find_peaks(model_it_css,height=1) #find peaks

fig = plt.figure(figsize=(24,48),constrained_layout=True)
outer = gridspec.GridSpec(1, 2, wspace=0.4)

for i in range(2):
    if i == 0: #first plot
        inner = gridspec.GridSpecFromSubplotSpec(1,1,
                                                 subplot_spec=outer[i])
        ax = plt.Subplot(fig, inner[0])
        ax.set_title('RF position for voxel %d of %s'%(index,hemi))
        ax.set_xlim([-analysis_params["max_eccen"],analysis_params["max_eccen"]])
        ax.set_ylim([-analysis_params["max_eccen"],analysis_params["max_eccen"]])
        ax.axvline(0, -15, 15, c='k', lw=0.25)
        ax.axhline(0, -15, 15, c='k', lw=0.25)
        ax.add_artist(plt.Circle((x_it_css,y_it_css), sigma_it_css, color='r',alpha=rsq_it_css))
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
            ax.imshow(prf_dm[:,:,sig_peaks[0][k]-3].T)
            ax.set_title('bar pass TR = %d'%(sig_peaks[0][k]-3))
            ax.set(adjustable='box-forced',aspect='equal')
            fig.add_subplot(ax)
            k += 1

fig.show()
fig.savefig(os.path.join(out_dir,'RF_voxel-%d_%s_shifted.png'%(index,hemi)), dpi=100,bbox_inches = 'tight')











