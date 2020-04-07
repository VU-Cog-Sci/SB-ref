# script to make motor flatmaps 
# to better visualize borders and add to overlay (fsaverage_meridians)
# so I can actually draw them

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


# define participant number and open json parameter file
if len(sys.argv)<2:	
    raise NameError('Please add subject number (ex:01) '	
                    'as 1st argument in the command line!')	

else:	
    sj = str(sys.argv[1]).zfill(2) #fill subject number with 0 in case user forgets	

    with open('analysis_params.json','r') as json_file:	
            analysis_params = json.load(json_file)	


# define paths
with_smooth = 'True'#'False'#analysis_params['with_smooth']

# dir to get soma contrasts
soma_dir = os.path.join(analysis_params['soma_outdir'],'new_fits','sub-{sj}'.format(sj=sj)) 

if with_smooth=='True':
    soma_dir = os.path.join(soma_dir,'smooth%d'%analysis_params['smooth_fwhm'])


## NOW DO SOMA PLOTS ###
rsq_threshold = 0 
z_threshold = 2.7 #analysis_params['z_threshold']

# load contrasts for different regions
face_contrast = np.load(os.path.join(soma_dir,'z_face_contrast_rsq-%.2f.npy' %(rsq_threshold)))
hand_contrast = np.load(os.path.join(soma_dir,'z_upper_limb_contrast_rsq-%.2f.npy' %(rsq_threshold)))

# plot different body areas
# but first threshold them (only show positive z-scores)
data_threshed_face = zthresh(face_contrast,threshold=z_threshold,side='above')
data_threshed_hand = zthresh(hand_contrast,threshold=z_threshold,side='above')

# mask to only show relevant voxels
rl_mask = np.array([True if np.isnan(val) else False for _,val in enumerate(data_threshed_hand)])


# all fingers in hand combined
LHfing_zscore = [] # load and append each finger z score in left hand list
RHfing_zscore = [] # load and append each finger z score in right hand list


print('Loading data for all fingers and appending in list')

for i in range(len(analysis_params['all_contrasts']['upper_limb'])//2):
    
    LHfing_zscore.append(np.load(os.path.join(soma_dir,'z_%s-all_lhand_contrast_thresh-%0.2f_rsq-%.2f.npy' 
                                 %(analysis_params['all_contrasts']['upper_limb'][i],z_threshold,rsq_threshold))))
    RHfing_zscore.append(np.load(os.path.join(soma_dir,'z_%s-all_rhand_contrast_thresh-%0.2f_rsq-%.2f.npy' 
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

LH_COM_4plot = LH_COM.copy()
LH_COM_4plot[rl_mask] = np.nan



images = {}

images['v_Lfingers'] = cortex.Vertex(LH_COM_4plot, 'fsaverage_meridians',
                           vmin=0, vmax=4,
                           cmap='rainbow_r')#costum colormap added to database

#cortex.quickshow(images['v_Lfingers'],with_curvature=True,with_sulci=True,with_colorbar=True)

# add to overlay
cortex.utils.add_roi(images['v_Lfingers'], name='Lhand_sub_%s_zthresh-%.2f'%(sj,z_threshold), open_inkscape=False)


# all fingers right hand combined ONLY in right hand region 
# (as defined by LvsR hand contrast values)

RH_COM_4plot = RH_COM.copy()
RH_COM_4plot[rl_mask] = np.nan

images['v_Rfingers'] = cortex.Vertex(RH_COM_4plot, 'fsaverage_meridians',
                           vmin=0, vmax=4,
                           cmap='rainbow_r')#costum colormap added to database

#cortex.quickshow(images['v_Rfingers'],with_curvature=True,with_sulci=True,with_colorbar=True)

# add to overlay
cortex.utils.add_roi(images['v_Rfingers'], name='Rhand_sub_%s_zthresh-%.2f'%(sj,z_threshold), open_inkscape=False)


# all individual face regions combined

allface_zscore = [] # load and append each face part z score in list

print('Loading data for each face part and appending in list')

for _,name in enumerate(analysis_params['all_contrasts']['face']):
    
    facedata = np.load(os.path.join(soma_dir,'z_%s-other_face_areas_contrast_thresh-%0.2f_rsq-%.2f.npy' %(name,z_threshold,rsq_threshold)))   
    allface_zscore.append(facedata)  

allface_zscore = np.array(allface_zscore)

# combine them all in same array

print('Computing center of mass for face elements %s' %(analysis_params['all_contrasts']['face']))
allface_COM , allface_avgzval = zsc_2_COM(allface_zscore)


# threshold left vs right, to only show relevant vertex 
# (i.e., where zscore is "significant", use it to mask face for plotting)
face_mask = np.array([True if np.isnan(val) else False for _,val in enumerate(data_threshed_face)])

allface_COM_4plot = allface_COM.copy()
allface_COM_4plot[face_mask] = np.nan



# 'eyebrows', 'eyes', 'mouth','tongue', , combined
images['v_facecombined'] = cortex.Vertex(allface_COM_4plot, 'fsaverage_meridians',
                           vmin=0, vmax=3,
                           cmap='J4') #costum colormap added to database


#cortex.quickshow(images['v_facecombined'],with_curvature=True,with_sulci=True,with_colorbar=True)

# add to overlay
cortex.utils.add_roi(images['v_facecombined'], name='Face_sub_%s_zthresh-%.2f'%(sj,z_threshold), open_inkscape=False)




