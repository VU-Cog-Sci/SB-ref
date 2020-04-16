# make prf and soma rsq maps
# combined - ideally to see which regions are motor and which are visual and where these overlap
# load estimates that were obtained previously in other scripts, so need to run individual task scripts first

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
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as matcm

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

# path to save figure
figure_out = os.path.join(analysis_params['derivatives'],'figures','multimodal','sub-{sj}'.format(sj=sj))
# dir to get soma contrasts
soma_dir = os.path.join(analysis_params['soma_outdir'],'new_fits','sub-{sj}'.format(sj=sj)) 

if with_smooth=='True':
    figure_out = os.path.join(figure_out,'smooth%d'%analysis_params['smooth_fwhm'])
    soma_dir = os.path.join(soma_dir,'smooth%d'%analysis_params['smooth_fwhm'])

if not os.path.exists(figure_out): # check if path to save figures exists
    os.makedirs(figure_out) 

### PRF PARAMS #######
# fit model to use (gauss or css)
fit_model = 'css' #analysis_params["fit_model"]

# if using estimates from iterative fit
iterative_fit = True #True

# total number of chunks that were fitted (per hemi)
total_chunks = analysis_params['total_chunks']
    
##

# filename for visual rsq
rsq_visual_filename = os.path.join(figure_out,'rsq_pRF_fitmodel-%s_itertivefit-%s.npy'%(fit_model,str(iterative_fit)))

if not os.path.isfile(rsq_visual_filename):
    # make list with subjects to append and use (or not)
    if sj == 'median':
        excl_subs = ['sub-03','sub-05','sub-07','sub-13']
    else:
        all_subs = ['01','02','03','04','05','07','08','09','11','12','13']
        excl_subs = ['sub-'+name for _,name in enumerate(all_subs) if name!=sj]
        
    # first append estimates (if median) or load if single sub
    estimates = append_pRFestimates(analysis_params['pRF_outdir'],
                                        model=fit_model,iterative=iterative_fit,exclude_subs=excl_subs,total_chunks=total_chunks)

    print('appended estimates for %s excluded %s'%(str(estimates['subs']),str(estimates['exclude_subs'])))

    rsq_visual = np.array(estimates['r2'])
    
    # now mask them according to screen dimensions
    masked_rsq = []

    for w in range(rsq_visual.shape[0]): # loop once if one subject, or for all subjects when sj 'all'

        subject = estimates['subs'][w]

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
                                       estimates['baseline'][w],rsq_visual[w],
                                       vert_lim_dva,hor_lim_dva,ns=estimates['ns'][w])

        masked_rsq.append(new_estimates['rsq'])

    # make median and save
    rsq_visual = np.nanmedian(np.array(rsq_visual),axis=0)
    
    if with_smooth == 'True':
        
        header_sub = '01' if sj=='median' else sj # to get header for smoothing
            
        rsq_visual = smooth_nparray(os.path.join(analysis_params['post_fmriprep_outdir'], 'prf', 'sub-{sj}'.format(sj=str(header_sub).zfill(2))),
                           rsq_visual,
                           figure_out,
                           'rsq',
                           rsq_visual_filename,
                           n_TR=83,
                           task='prf',
                           file_extension='_cropped_sg_psc.func.gii',
                           sub_mesh='fsaverage',
                           smooth_fwhm=analysis_params['smooth_fwhm'])
        
    
    np.save(rsq_visual_filename,rsq_visual)
    
else:
    print('loading visual rsq from %s'%rsq_visual_filename)
    rsq_visual = np.load(rsq_visual_filename,allow_pickle=True)


# load soma rsq
print('load %s'%os.path.join(soma_dir,'rsq.npy'))
rsq_motor = np.load(os.path.join(soma_dir,'rsq.npy'))


# do this to replace nans with 0s, so flatmaps look nicer
rsq_visual = np.array([x if np.isnan(x)==False else 0 for _,x in enumerate(rsq_visual)])
rsq_motor = np.array([x if np.isnan(x)==False else 0 for _,x in enumerate(rsq_motor)])

# normalize RSQ 
rsq_visual_norm = normalize(rsq_visual) 
rsq_motor_norm = normalize(rsq_motor)


# create costume colormp red blue
n_bins = 256
col2D_name = os.path.splitext(os.path.split(make_2D_colormap(rgb_color='101',bins=n_bins))[-1])[0]
print('created costum colormap %s'%col2D_name)


# make flatmaps of the above distributions
print('making flatmaps')

images = {}

images['rsq_visual_norm'] = cortex.Vertex(rsq_visual_norm,'fsaverage_gross',
                           vmin=0, vmax=1,
                           cmap='Reds')
#cortex.quickshow(images['rsq_visual_norm'],with_curvature=True,with_sulci=True)
filename = os.path.join(figure_out,'flatmap_space-fsaverage_type-rsquared-normalized_visual.svg')
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['rsq_visual_norm'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True)

images['rsq_motor_norm'] = cortex.Vertex(rsq_motor_norm,'fsaverage_gross',
                           vmin=0, vmax=1,
                           cmap='Blues')
#cortex.quickshow(images['rsq_motor_norm'],with_curvature=True,with_sulci=True)
filename = os.path.join(figure_out,'flatmap_space-fsaverage_type-rsquared-normalized_motor.svg')
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['rsq_motor_norm'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True)


images['rsq_combined'] = cortex.Vertex2D(rsq_visual_norm,rsq_motor_norm, 
                            subject='fsaverage_gross',
                            vmin=0.125, vmax=0.2,
                            vmin2=0.2,vmax2=0.6,
                            cmap=col2D_name)#'PU_RdBu_covar')
#cortex.quickshow(images['rsq_combined'],recache=True,with_curvature=True,with_sulci=True,with_roi=False,height=2048)
filename = os.path.join(figure_out,'flatmap_space-fsaverage_type-rsquared-normalized_combined_bins-%d.svg'%n_bins)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['rsq_combined'], recache=True,with_roi=True,with_colorbar=False,with_curvature=True,with_sulci=True,height=2048)


# Plot a flatmap with the data projected onto the surface
# Highlight the curvature and which cutout to be displayed

# Name of a sub-layer of the 'cutouts' layer in overlays.svg file
cutout_name = 'zoom_roi_left'
_ = cortex.quickflat.make_figure(images['rsq_combined'],
                                 with_curvature=True,
                                 with_sulci=True,
                                 with_roi=False,
                                 with_colorbar=False,
                                 cutout=cutout_name,height=2048)

filename = os.path.join(figure_out,cutout_name+'_space-fsaverage_type-rsquared-normalized_combined_bins-%d.svg'%n_bins)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['rsq_combined'], recache=True,with_colorbar=False,
                              cutout=cutout_name,with_curvature=True,with_sulci=True,with_roi=False,height=2048)


# Name of a sub-layer of the 'cutouts' layer in overlays.svg file
cutout_name = 'zoom_roi_right'
_ = cortex.quickflat.make_figure(images['rsq_combined'],
                                 with_curvature=True,
                                 with_sulci=True,
                                 with_roi=False,
                                 with_colorbar=False,
                                 cutout=cutout_name,height=2048)
filename = os.path.join(figure_out,cutout_name+'_space-fsaverage_type-rsquared-normalized_combined_bins-%d.svg'%n_bins)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['rsq_combined'], recache=True,with_colorbar=False,
                              cutout=cutout_name,with_curvature=True,with_sulci=True,with_roi=False,height=2048)



### POLAR ANGLES MAPS #####
# now load estimates to make polar angle
# should have run make_PA4drawing script first

prf_path = os.path.join(analysis_params['derivatives'],'figures','prf','final_fig',fit_model)

if iterative_fit==True:
    prf_path = os.path.join(prf_path,'iterative','sub-{sj}'.format(sj=sj))
else:
    prf_path = os.path.join(prf_path,'grid','sub-{sj}'.format(sj=sj))
    
if with_smooth=='True':
    prf_path = os.path.join(prf_path,'smooth%d'%analysis_params['smooth_fwhm'])
    
# filenames
xx_filename = os.path.join(prf_path,'xx_pRF_fitmodel-%s_itertivefit-%s.npy'%(fit_model,str(iterative_fit)))
yy_filename = os.path.join(prf_path,'yy_pRF_fitmodel-%s_itertivefit-%s.npy'%(fit_model,str(iterative_fit)))

# load
print('loading visual xx from %s'%xx_filename)
masked_xx = np.load(xx_filename,allow_pickle=True)

print('loading visual yy from %s'%yy_filename)
masked_yy = np.load(yy_filename,allow_pickle=True)

rsq_threshold = 0.14

complex_location = masked_xx + masked_yy * 1j
masked_polar_angle = np.angle(complex_location)
masked_eccentricity = np.abs(complex_location)    

images = {}

# normalize polar angles to have values in circle between 0 and 1
masked_polar_ang_norm = (masked_polar_angle + np.pi) / (np.pi * 2.0)

##### make PA flatmaps with non uniform color wheel ######
# shift radians in order to overrepresent red color
# useful to make NON-REPRESENTED retinotopic hemifield per hemisphere red
# then easier to define borders

# create HSV array, with PA values (-pi to pi) that were obtained from estimates
# saturation wieghted by a shifted distribution of RSQ (better visualization)
# value bolean (if I don't give it an rsq threshold then it's always 1)

hsv_angle = []
hsv_angle = np.ones((len(rsq_visual), 3))
hsv_angle[:, 0] = masked_polar_angle.copy()
#hsv_angle[:, 1] = np.clip(rsq_visual / np.nanmax(rsq_visual) * 3, 0, 1)
hsv_angle[:, 2] = rsq_visual > rsq_threshold #np.clip(rsq_visual / np.nanmax(rsq_visual) * 3, 0, 1)#rsq_visual > 0.12 #rsq_threshold

# get mid vertex index (diving hemispheres)
left_index = cortex.db.get_surfinfo('fsaverage').left.shape[0] 

### take angles from LH (thus RVF)##
angle_ = hsv_angle[:left_index, 0].copy()
angle_thresh = 3*np.pi/4 #value upon which to make it red for this hemifield (above it or below -angle will be red)

#normalized angles, between 0 and 1
hsv_angle[:left_index, 0] = np.clip((angle_ + angle_thresh)/(2*angle_thresh), 0, 1)

### take angles from RH (thus LVF) ##
angle_ = -hsv_angle[left_index:, 0].copy() # ATENÇÃO -> minus sign to flip angles vertically (then order of colors same for both hemispheres)

# sum 2pi to originally positive angles (now our trig circle goes from pi to 2pi to pi again, all positive)
angle_[hsv_angle[left_index:, 0] > 0] += 2 * np.pi

#normalized angles, between 0 and 1
angle_ = np.clip((angle_ + (angle_thresh-np.pi))/(2*angle_thresh), 0, 1) # ATENÇÃO -> we subtract -pi to angle thresh because now we want to rotate the whole thing -180 degrees

hsv_angle[left_index:, 0] = angle_.copy()
rgb_angle = []
rgb_angle = colors.hsv_to_rgb(hsv_angle)

# make alpha same as saturation, reduces clutter
alpha_angle = hsv_angle[:, 2]

images['angle_half_hemi'] = cortex.VertexRGB(rgb_angle[:, 0], rgb_angle[:, 1], rgb_angle[:, 2],
                                   alpha=alpha_angle,
                                   subject='fsaverage_meridians')
#cortex.quickshow(images['angle_half_hemi'],with_curvature=True,with_sulci=True,with_colorbar=False)

filename = os.path.join(figure_out,'flatmap_space-fsaverage_rsq-%0.2f_type-polar_angle_half_colorwheel.svg' %rsq_threshold)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['angle_half_hemi'], recache=True,with_colorbar=False,with_curvature=True,with_sulci=True)


# Plot a flatmap with the data projected onto the surface
# Highlight the curvature and which cutout to be displayed

# Name of a sub-layer of the 'cutouts' layer in overlays.svg file
cutout_name = 'zoom_roi_left'
_ = cortex.quickflat.make_figure(images['angle_half_hemi'],
                                 with_curvature=True,
                                 with_sulci=True,
                                 with_roi=False,
                                 with_colorbar=False,
                                 cutout=cutout_name,height=2048)

filename = os.path.join(figure_out,cutout_name+'_space-fsaverage_type-PA_bins-%d.svg'%n_bins)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['angle_half_hemi'], recache=True,with_colorbar=False,
                              cutout=cutout_name,with_curvature=True,with_sulci=True,with_roi=False,height=2048)

# Name of a sub-layer of the 'cutouts' layer in overlays.svg file
cutout_name = 'zoom_roi_right'
_ = cortex.quickflat.make_figure(images['angle_half_hemi'],
                                 with_curvature=True,
                                 with_sulci=True,
                                 with_roi=False,
                                 with_colorbar=False,
                                 cutout=cutout_name,height=2048)
filename = os.path.join(figure_out,cutout_name+'_space-fsaverage_type-PA_bins-%d.svg'%n_bins)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['angle_half_hemi'], recache=True,with_colorbar=False,
                              cutout=cutout_name,with_curvature=True,with_sulci=True,with_roi=False,height=2048)


## NOW DO SOMA PLOTS ###
rsq_threshold = 0 
z_threshold = analysis_params['z_threshold'] #2.7 #

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

images['v_Lfingers'] = cortex.Vertex(LH_COM_4plot, 'fsaverage_meridians',
                           vmin=0, vmax=4,
                           cmap='rainbow_r')#costum colormap added to database

#cortex.quickshow(images['v_Lfingers'],with_curvature=True,with_sulci=True,with_colorbar=True)
filename = os.path.join(figure_out,'flatmap_space-fsaverage_rsq-%0.2f_type-LH-fingers1-5.svg' %(rsq_threshold))
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_Lfingers'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True)


# Name of a sub-layer of the 'cutouts' layer in overlays.svg file
cutout_name = 'zoom_roi_right'
_ = cortex.quickflat.make_figure(images['v_Lfingers'],
                                 with_curvature=True,
                                 with_sulci=True,
                                 with_roi=False,
                                 with_colorbar=False,
                                 cutout=cutout_name,height=2048)
filename = os.path.join(figure_out,cutout_name+'_space-fsaverage_type-LeftHand_bins-%d.svg'%n_bins)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_Lfingers'], recache=True,with_colorbar=False,
                              cutout=cutout_name,with_curvature=True,with_sulci=True,with_roi=False,height=2048)

# all fingers right hand combined ONLY in right hand region 
# (as defined by LvsR hand contrast values)

RH_COM_4plot = RH_COM.copy()
RH_COM_4plot[rl_mask] = np.nan

images['v_Rfingers'] = cortex.Vertex(RH_COM_4plot, 'fsaverage_meridians',
                           vmin=0, vmax=4,
                           cmap='rainbow_r')#costum colormap added to database

#cortex.quickshow(images['v_Rfingers'],with_curvature=True,with_sulci=True,with_colorbar=True)
filename = os.path.join(figure_out,'flatmap_space-fsaverage_rsq-%0.2f_type-RH-fingers1-5.svg' %(rsq_threshold))
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_Rfingers'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True)

# Name of a sub-layer of the 'cutouts' layer in overlays.svg file
cutout_name = 'zoom_roi_left'
_ = cortex.quickflat.make_figure(images['v_Rfingers'],
                                 with_curvature=True,
                                 with_sulci=True,
                                 with_roi=False,
                                 with_colorbar=False,
                                 cutout=cutout_name,height=2048)
filename = os.path.join(figure_out,cutout_name+'_space-fsaverage_type-RightHand_bins-%d.svg'%n_bins)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_Rfingers'], recache=True,with_colorbar=False,
                              cutout=cutout_name,with_curvature=True,with_sulci=True,with_roi=False,height=2048)

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
filename = os.path.join(figure_out,'flatmap_space-fsaverage_rsq-%0.2f_zscore-%.2f_type-eyebrows-eyes-mouth-tongue.svg' %(rsq_threshold,z_threshold))
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_facecombined'], recache=False,with_colorbar=True,with_curvature=True,with_sulci=True)


# Name of a sub-layer of the 'cutouts' layer in overlays.svg file
cutout_name = 'zoom_roi_right'
_ = cortex.quickflat.make_figure(images['v_facecombined'],
                                 with_curvature=True,
                                 with_sulci=True,
                                 with_roi=False,
                                 with_colorbar=False,
                                 cutout=cutout_name,height=2048)
filename = os.path.join(figure_out,cutout_name+'_space-fsaverage_type-Face_RH_bins-%d.svg'%n_bins)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_facecombined'], recache=True,with_colorbar=False,
                              cutout=cutout_name,with_curvature=True,with_sulci=True,with_roi=False,height=2048)


# Name of a sub-layer of the 'cutouts' layer in overlays.svg file
cutout_name = 'zoom_roi_left'
_ = cortex.quickflat.make_figure(images['v_facecombined'],
                                 with_curvature=True,
                                 with_sulci=True,
                                 with_roi=False,
                                 with_colorbar=False,
                                 cutout=cutout_name,height=2048)
filename = os.path.join(figure_out,cutout_name+'_space-fsaverage_type-Face_LH_bins-%d.svg'%n_bins)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_facecombined'], recache=True,with_colorbar=False,
                              cutout=cutout_name,with_curvature=True,with_sulci=True,with_roi=False,height=2048)


## FOR FIGURE ##############################

# Name of a sub-layer of the 'cutouts' layer in overlays.svg file
cutout_name = 'zoom_face_right'
_ = cortex.quickflat.make_figure(images['v_facecombined'],
                                 with_curvature=True,
                                 with_sulci=True,
                                 with_roi=False,
                                 with_colorbar=False,
                                 cutout=cutout_name,height=2048)
filename = os.path.join(figure_out,cutout_name+'_space-fsaverage_type-Face_zoomed_RH_bins-%d.svg'%n_bins)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_facecombined'], recache=True,with_colorbar=False,
                              cutout=cutout_name,with_curvature=True,with_sulci=True,with_roi=False,height=2048)


# Name of a sub-layer of the 'cutouts' layer in overlays.svg file
cutout_name = 'zoom_face_left'
_ = cortex.quickflat.make_figure(images['v_facecombined'],
                                 with_curvature=True,
                                 with_sulci=True,
                                 with_roi=False,
                                 with_colorbar=False,
                                 cutout=cutout_name,height=2048)
filename = os.path.join(figure_out,cutout_name+'_space-fsaverage_type-Face_zoomed_LH_bins-%d.svg'%n_bins)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_facecombined'], recache=True,with_colorbar=False,
                              cutout=cutout_name,with_curvature=True,with_sulci=True,with_roi=False,height=2048)


# Name of a sub-layer of the 'cutouts' layer in overlays.svg file
cutout_name = 'zoom_hand_right'
_ = cortex.quickflat.make_figure(images['v_Lfingers'],
                                 with_curvature=True,
                                 with_sulci=True,
                                 with_roi=False,
                                 with_colorbar=False,
                                 cutout=cutout_name,height=2048)
filename = os.path.join(figure_out,cutout_name+'_space-fsaverage_type-LeftHand_zoomed_bins-%d.svg'%n_bins)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_Lfingers'], recache=True,with_colorbar=False,
                              cutout=cutout_name,with_curvature=True,with_sulci=True,with_roi=False,height=2048)

# Name of a sub-layer of the 'cutouts' layer in overlays.svg file
cutout_name = 'zoom_hand_left'
_ = cortex.quickflat.make_figure(images['v_Rfingers'],
                                 with_curvature=True,
                                 with_sulci=True,
                                 with_roi=False,
                                 with_colorbar=False,
                                 cutout=cutout_name,height=2048)
filename = os.path.join(figure_out,cutout_name+'_space-fsaverage_type-RightHand_zoomed_bins-%d.svg'%n_bins)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_Rfingers'], recache=True,with_colorbar=False,
                              cutout=cutout_name,with_curvature=True,with_sulci=True,with_roi=False,height=2048)


print('Done!')




