#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Th Jun 6 11:13:11 2019

@author: inesverissimo

visualize median subject in vertex space
"""

import os, json
import sys, glob
import numpy as np
import cortex
import matplotlib.colors as colors
from utils import *


# open json parameter file

with open('analysis_params.json','r') as json_file:	
        analysis_params = json.load(json_file)	
    
z_threshold = analysis_params['z_threshold']

# load and compute median contrast
contrast_dir = [os.path.join(analysis_params['soma_outdir'], f) for f in os.listdir(analysis_params['soma_outdir'])]
contrast_dir.sort()

for idx,sub in enumerate(contrast_dir):
    sub_dir = os.path.join(sub,'run-median')
    
    if idx == 0:
        ## group different body areas
        face_zscore = np.load(os.path.join(sub_dir,'z_face_contrast.npy'))
        upper_zscore = np.load(os.path.join(sub_dir,'z_upper_limb_contrast.npy'))
        lower_zscore = np.load(os.path.join(sub_dir,'z_lower_limb_contrast.npy'))
        
        ## Right vs left        
        RLupper_zscore = np.load(os.path.join(sub_dir,'z_right-left_hand_contrast.npy'))
        RLlower_zscore = np.load(os.path.join(sub_dir,'z_right-left_leg_contrast.npy'))
        
    else:
        face_zscore = np.vstack((face_zscore,np.load(os.path.join(sub_dir,'z_face_contrast.npy'))))
        upper_zscore = np.vstack((upper_zscore,np.load(os.path.join(sub_dir,'z_upper_limb_contrast.npy'))))
        lower_zscore = np.vstack((lower_zscore,np.load(os.path.join(sub_dir,'z_lower_limb_contrast.npy'))))
        
        RLupper_zscore = np.vstack((RLupper_zscore,np.load(os.path.join(sub_dir,'z_right-left_hand_contrast.npy'))))
        RLlower_zscore = np.vstack((RLlower_zscore,np.load(os.path.join(sub_dir,'z_right-left_leg_contrast.npy'))))
        
face_median_zscore = np.median(face_zscore,axis=0)
upper_median_zscore = np.median(upper_zscore,axis=0)
lower_median_zscore = np.median(lower_zscore,axis=0)

RLupper_median_zscore = np.median(RLupper_zscore,axis=0)
RLlower_median_zscore = np.median(RLlower_zscore,axis=0)

# threshold them
data_threshed_median_face = zthresh(face_median_zscore,z_threshold,side='both')
data_threshed_median_upper = zthresh(upper_median_zscore,z_threshold,side='both')
data_threshed_median_lower = zthresh(lower_median_zscore,z_threshold,side='both')

data_threshed_median_RLhand=zthresh(RLupper_median_zscore,z_threshold,side='both')
data_threshed_median_RLleg=zthresh(RLlower_median_zscore,z_threshold,side='both')


# combine 3 body part maps, threshold values
combined_median_zvals = np.array((face_median_zscore,upper_median_zscore,lower_median_zscore))

soma_median_labels, soma_median_zval = winner_takes_all(combined_median_zvals,analysis_params['all_contrasts'],z_threshold,side='above')

# now do same for fingers 
for idx,sub in enumerate(contrast_dir):
    sub_dir = os.path.join(sub,'run-median')
    
    if idx == 0:
        # all fingers in hand combined
        LHfing_zscore = [] # load and append each finger z score in left hand list
        RHfing_zscore = [] # load and append each finger z score in right hand list

        for i in range(len(analysis_params['all_contrasts']['upper_limb'])//2):

            Ldata = np.load(os.path.join(sub_dir,'z_%s-all_lhand_contrast.npy' %(analysis_params['all_contrasts']['upper_limb'][i])))
            Rdata = np.load(os.path.join(sub_dir,'z_%s-all_rhand_contrast.npy' %(analysis_params['all_contrasts']['upper_limb'][i+5])))

            LHfing_zscore.append(Ldata)  
            RHfing_zscore.append(Rdata)

        LHfing_zscore = np.array(LHfing_zscore)
        LHfing_zscore_newax = LHfing_zscore[:,:, np.newaxis]
        RHfing_zscore = np.array(RHfing_zscore)
        RHfing_zscore_newax = RHfing_zscore[:,:, np.newaxis]
        
    else:
        
        LHfing_zscore = [] # load and append each finger z score in left hand list
        RHfing_zscore = [] # load and append each finger z score in right hand list

        for i in range(len(analysis_params['all_contrasts']['upper_limb'])//2):

            Ldata = np.load(os.path.join(sub_dir,'z_%s-all_lhand_contrast.npy' %(analysis_params['all_contrasts']['upper_limb'][i])))
            Rdata = np.load(os.path.join(sub_dir,'z_%s-all_rhand_contrast.npy' %(analysis_params['all_contrasts']['upper_limb'][i+5])))

            LHfing_zscore.append(Ldata)  
            RHfing_zscore.append(Rdata)

        LHfing_zscore = np.array(LHfing_zscore)
        RHfing_zscore = np.array(RHfing_zscore)
        
        LHfing_zscore_newax = np.concatenate((LHfing_zscore_newax,LHfing_zscore[:,:, np.newaxis]),axis=2)
        RHfing_zscore_newax = np.concatenate((RHfing_zscore_newax,RHfing_zscore[:,:, np.newaxis]),axis=2)
        
        
LHfing_median_zscore = np.median(LHfing_zscore_newax,axis=2)
RHfing_median_zscore = np.median(RHfing_zscore_newax,axis=2)


LH_median_labels, LH_median_zval = winner_takes_all(LHfing_median_zscore,
                                      analysis_params['all_contrasts']['upper_limb'][:5],z_threshold,side='above')

RH_median_labels, RH_median_zval = winner_takes_all(RHfing_median_zscore,
                                      analysis_params['all_contrasts']['upper_limb'][5:],z_threshold,side='above')


## define ROI for each hand, to plot finger z score maps in relevant areas

# by using Left vs Right hand maps
LH_median_roiLR = np.zeros(data_threshed_median_RLhand.shape) # set at 0 whatever is outside thresh
RH_median_roiLR = np.zeros(data_threshed_median_RLhand.shape) # set at 0 whatever is outside thresh

for i,zsc in enumerate(data_threshed_median_RLhand): # loop over thresholded RvsL hand zscores
    if zsc < 0: # negative z-scores = left hand
        LH_median_roiLR[i]=LH_median_zval[i]
    elif zsc > 0: # positive z-scores = right hand
        RH_median_roiLR[i]=RH_median_zval[i]


# now do same for face  

for idx,sub in enumerate(contrast_dir):
    sub_dir = os.path.join(sub,'run-median')
    
    if idx == 0:
        allface_zscore = [] # load and append each face part z score in list

        for _,name in enumerate(analysis_params['all_contrasts']['face']):

            facedata = np.load(os.path.join(sub_dir,'z_%s-other_face_areas_contrast.npy' %(name)))   
            allface_zscore.append(facedata)  

        allface_zscore = np.array(allface_zscore)
        allface_zscore_newax = allface_zscore[:,:, np.newaxis]
        
    else:
        
        allface_zscore = [] # load and append each face part z score in list

        for _,name in enumerate(analysis_params['all_contrasts']['face']):

            facedata = np.load(os.path.join(sub_dir,'z_%s-other_face_areas_contrast.npy' %(name)))   
            allface_zscore.append(facedata)  

        allface_zscore = np.array(allface_zscore)
 
        allface_zscore_newax = np.concatenate((allface_zscore_newax,allface_zscore[:,:, np.newaxis]),axis=2)
        
        
allface_median_zscore = np.median(allface_zscore_newax,axis=2)

# combine them all in same array, winner takes all
allface_median_labels, allface_median_zval = winner_takes_all(allface_median_zscore,
                                      analysis_params['all_contrasts']['face'],z_threshold,side='above')

# define ROI by using face map, 
# to plot face part z score maps in relevant areas
face_median_roiF = np.zeros(data_threshed_median_face.shape) # set at 0 whatever is outside thresh   

for i,zsc in enumerate(data_threshed_median_face):
    if zsc > 0: # ROI only accounts for positive z score areas
        face_median_roiF[i]=allface_median_zval[i]



## create flatmaps for different parameters and save png
images = {}

# vertex for face vs all others
images['v_median_face'] = cortex.Vertex(data_threshed_median_face.T, 'fsaverage',
                           vmin=-5, vmax=5,
                           cmap='BuBkRd')

# vertex for upper limb vs all others
images['v_median_upper'] = cortex.Vertex(data_threshed_median_upper.T, 'fsaverage',
                           vmin=-5, vmax=5,
                           cmap='BuBkRd')

# vertex for lower limb vs all others
images['v_median_lower'] = cortex.Vertex(data_threshed_median_lower.T, 'fsaverage',
                           vmin=-5, vmax=5,
                           cmap='BuBkRd')

# all somas combined
images['v_median_combined'] = cortex.Vertex2D(soma_median_labels.T, soma_median_zval.T, 'fsaverage',
                           vmin=0, vmax=1,
                           vmin2=z_threshold, vmax2=5, cmap='autumn_alpha')#BROYG_2D')#'my_autumn')

# vertex for right vs left hand
images['rl_median_upper'] = cortex.Vertex(data_threshed_median_RLhand.T, 'fsaverage',
                           vmin=-5, vmax=5,
                           cmap='bwr')

# vertex for right vs left leg
images['rl_median_lower'] = cortex.Vertex(data_threshed_median_RLleg.T, 'fsaverage',
                           vmin=-5, vmax=5,
                           cmap='bwr')

# all finger right hand combined
images['v_median_Rfingers'] = cortex.Vertex2D(RH_median_labels.T, RH_median_zval.T, 'fsaverage',
                           vmin=0, vmax=1,
                           vmin2=z_threshold, vmax2=5, cmap='BROYG_2D')#BROYG_2D')#'my_autumn')

# all finger left hand combined
images['v_median_Lfingers'] = cortex.Vertex2D(LH_median_labels.T, LH_median_zval.T, 'fsaverage',
                           vmin=0, vmax=1,
                           vmin2=z_threshold, vmax2=5, cmap='BROYG_2D')#BROYG_2D')#'my_autumn')

# all fingers left hand combined ONLY in left hand region 
# (as defined by LvsR hand contrast values)
images['v_median_LfingersROILR'] = cortex.Vertex2D(LH_median_labels.T, LH_median_roiLR.T, 'fsaverage',
                           vmin=0, vmax=1,
                           vmin2=z_threshold, vmax2=5, cmap='BROYG_2D')#BROYG_2D')#'my_autumn')

# all fingers right hand combined ONLY in right hand region 
# (as defined by LvsR hand contrast values)
images['v_median_RfingersROILR'] = cortex.Vertex2D(RH_median_labels.T, RH_median_roiLR.T, 'fsaverage',
                           vmin=0, vmax=1,
                           vmin2=z_threshold, vmax2=5, cmap='BROYG_2D')#BROYG_2D')#'my_autumn')

# 'eyebrows', 'eyes', 'tongue', 'mouth', combined
images['v_median_facecombined'] = cortex.Vertex2D(allface_median_labels.T, allface_median_zval.T, 'fsaverage',
                           vmin=0, vmax=1,
                           vmin2=z_threshold, vmax2=5, cmap='BROYG4col_2D') #costum colormap added to database

images['v_median_facecombinedROIF'] = cortex.Vertex2D(allface_median_labels.T, face_median_roiF.T, 'fsaverage',
                           vmin=0, vmax=1,
                           vmin2=z_threshold, vmax2=5, cmap='BROYG4col_2D')#BROYG_2D')#'my_autumn')



flatmap_median_out = os.path.join(analysis_params['cortex_dir'],'sub-median','flatmaps')
if not os.path.exists(flatmap_median_out): # check if path for outputs exist
        os.makedirs(flatmap_median_out)       # if not create it


# Save this flatmap

filename = os.path.join(flatmap_median_out,'flatmap_space-fsaverage_zthresh-%0.2f_type-faceVSall.png' %z_threshold)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_median_face'], recache=True,with_colorbar=True,with_curvature=True)

# Save this flatmap
filename = os.path.join(flatmap_median_out,'flatmap_space-fsaverage_zthresh-%0.2f_type-upperVSall.png' %z_threshold)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_median_upper'], recache=True,with_colorbar=True,with_curvature=True)

# Save this flatmap
filename = os.path.join(flatmap_median_out,'flatmap_space-fsaverage_zthresh-%0.2f_type-lowerVSall.png' %z_threshold)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_median_lower'], recache=True,with_colorbar=True,with_curvature=True)

# Save this flatmap
filename = os.path.join(flatmap_median_out,'flatmap_space-fsaverage_zthresh-%0.2f_type-FULcombined.png' %z_threshold)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_median_combined'], recache=True,with_colorbar=True,with_curvature=True)

# Save this flatmap
filename = os.path.join(flatmap_median_out,'flatmap_space-fsaverage_zthresh-%0.2f_type-rightVSleftHAND.png' %z_threshold)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['rl_median_upper'], recache=True,with_colorbar=True,with_curvature=True)

# Save this flatmap
filename = os.path.join(flatmap_median_out,'flatmap_space-fsaverage_zthresh-%0.2f_type-rightVSleftLEG.png' %z_threshold)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['rl_median_lower'], recache=True,with_colorbar=True,with_curvature=True)


# Save this flatmap
filename = os.path.join(flatmap_median_out,'flatmap_space-fsaverage_zthresh-%0.2f_type-RHfing.png' %z_threshold)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_median_Rfingers'], recache=True,with_colorbar=True,with_curvature=True)

# Save this flatmap
filename = os.path.join(flatmap_median_out,'flatmap_space-fsaverage_zthresh-%0.2f_type-LHfing.png' %z_threshold)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_median_Lfingers'], recache=True,with_colorbar=True,with_curvature=True)

# Save this flatmap
filename = os.path.join(flatmap_median_out,'flatmap_space-fsaverage_zthresh-%0.2f_type-RHfing_ROI-LvsRH.png' %z_threshold)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_median_RfingersROILR'], recache=True,with_colorbar=True,with_curvature=True)

# Save this flatmap
filename = os.path.join(flatmap_median_out,'flatmap_space-fsaverage_zthresh-%0.2f_type-LHfing_ROI-LvsRH.png' %z_threshold)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_median_LfingersROILR'], recache=True,with_colorbar=True,with_curvature=True)


# Save this flatmap
filename = os.path.join(flatmap_median_out,'flatmap_space-fsaverage_zthresh-%0.2f_type-facecombined.png' %z_threshold)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_median_facecombined'], recache=True,with_colorbar=True,with_curvature=True)

# Save this flatmap
filename = os.path.join(flatmap_median_out,'flatmap_space-fsaverage_zthresh-%0.2f_type-facecombined_ROI-faceVsall.png' %z_threshold)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_median_facecombinedROIF'], recache=True,with_colorbar=True,with_curvature=True)

