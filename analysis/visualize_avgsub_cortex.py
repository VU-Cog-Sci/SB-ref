



import os, json
import sys, glob
import numpy as np
import cortex
import matplotlib.colors as colors
from utils import *

with open('analysis_params.json','r') as json_file:	
        analysis_params = json.load(json_file)	

rsq_threshold = analysis_params['rsq_threshold']
sj = 'median'

flatmap_out = os.path.join(analysis_params['cortex_dir'],'sub-median'.format(sj=sj),'flatmaps')
if not os.path.exists(flatmap_out): # check if path for outputs exist
        os.makedirs(flatmap_out)       # if not create it

### PRF #####

# load all subject estimates and compute weighted average to plot

allsubs = os.listdir(analysis_params['pRF_outdir'])
allsubs.sort()

sub_list = []
rsq = []
xx = []
yy = []
size = []
baseline = []
beta = []

for idx,est in enumerate(allsubs):
    sub_list.append(os.path.join(analysis_params['pRF_outdir'],est,'run-median'))
    
    estimates_list = [x for x in os.listdir(sub_list[idx]) if x.endswith('.estimates.npz') ]
    estimates_list.sort() #sort to make sure pRFs not flipped
    
    lhemi_est = np.load(os.path.join(sub_list[idx], estimates_list[0]))
    rhemi_est = np.load(os.path.join(sub_list[idx], estimates_list[1]))
    
    # concatenate r2 and parameteres, to later visualize whole brain (appending left and right together)
    rsq.append(np.concatenate((lhemi_est['r2'],rhemi_est['r2'])))
    
    xx.append(np.concatenate((lhemi_est['x'],rhemi_est['x'])))
  
    yy.append(np.concatenate((lhemi_est['y'],rhemi_est['y'])))

    size.append(np.concatenate((lhemi_est['size'],rhemi_est['size'])))
    baseline.append(np.concatenate((lhemi_est['baseline'],rhemi_est['baseline'])))
    beta.append(np.concatenate((lhemi_est['betas'],rhemi_est['betas'])))


med_rsq = np.median(np.array(rsq),axis=0) # median rsq

# make rsq mask where 0 is nan (because of 0 divisions in average)
rsq_mask = rsq[:]
for i,arr in enumerate(rsq):
    rsq_mask[i][arr==0] = np.nan

med_xx = np.average(np.array(xx),axis=0,weights=np.array(rsq_mask))
med_yy = np.average(np.array(yy),axis=0,weights=np.array(rsq_mask))

med_size = np.average(np.array(size),axis=0,weights=np.array(rsq_mask))

med_baseline = np.average(np.array(baseline),axis=0,weights=np.array(rsq_mask))
med_beta = np.average(np.array(beta),axis=0,weights=np.array(rsq_mask))


complex_location = med_xx + med_yy * 1j
polar_angle = np.angle(complex_location)
eccentricity = np.abs(complex_location)

# normalize polar angles to have values in circle between 0 and 1 
polar_ang_norm = (polar_angle + np.pi) / (np.pi * 2.0)

# use "resto da divisão" so that 1 == 0 (because they overlapp in circle)
# why have an offset?
angle_offset = 0.1
polar_ang_norm = np.fmod(polar_ang_norm+angle_offset, 1.0)

# convert angles to colors, using correlations as weights
hsv = np.zeros(list(polar_ang_norm.shape) + [3])
hsv[..., 0] = polar_ang_norm # different hue value for each angle
hsv[..., 1] = (np.array(med_rsq) > rsq_threshold).astype(float)#  np.ones_like(rsq) # saturation weighted by rsq
hsv[..., 2] = (np.array(med_rsq) > rsq_threshold).astype(float) # value weighted by rsq

# convert hsv values of np array to rgb values (values assumed to be in range [0, 1])
rgb = colors.hsv_to_rgb(hsv)

# define alpha channel - which specifies the opacity for a color
# 0 = transparent = values with rsq below thresh and 1 = opaque = values above thresh
alpha_mask = (np.array(med_rsq) <= rsq_threshold).T #why transpose? because of orientation of pycortex volume?
alpha = np.ones(alpha_mask.shape)
alpha[alpha_mask] = 0


images = {}
# vertex for polar angles
images['polar'] = cortex.VertexRGB(rgb[..., 0].T, 
                               rgb[..., 1].T, 
                               rgb[..., 2].T, 
                               subject='fsaverage') #, alpha=alpha

# vertex for rsq
images['rsq'] = cortex.Vertex2D(med_rsq.T, alpha, 'fsaverage',
                         vmin=0, vmax=1.0,
                         vmin2=0, vmax2=1.0, cmap='Reds_cov')

images['ecc'] = cortex.Vertex2D(eccentricity.T, alpha*10, 'fsaverage',
                             vmin=0, vmax=10,
                             vmin2=0, vmax2=1.0, cmap='BROYG_2D')

# vertex for size
images['size'] = cortex.dataset.Vertex2D(med_size.T, alpha*10, 'fsaverage',
                         vmin=0, vmax=10,
                         vmin2=0, vmax2=1.0, cmap='BROYG_2D')

# vertex for betas (amplitude?)
images['beta'] = cortex.Vertex2D(med_beta.T, med_rsq.T, 'fsaverage',
                         vmin=-2.5, vmax=2.5,
                         vmin2=rsq_threshold, vmax2=1.0, cmap='RdBu_r_alpha')

# vertex for baseline
images['baseline'] = cortex.Vertex2D(med_baseline.T, med_rsq.T, 'fsaverage',
                         vmin=-1, vmax=1,
                         vmin2=rsq_threshold, vmax2=1.0, cmap='RdBu_r_alpha')


# create flatmaps for different parameters and save png

# Save this flatmap
filename = os.path.join(flatmap_out,'flatmap_space-fsaverage_rsq-%0.2f_type-polar_angle.png' %rsq_threshold)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['polar'], recache=True,with_colorbar=False,with_curvature=True)

# Save this flatmap
filename = os.path.join(flatmap_out,'flatmap_space-fsaverage_rsq-%0.2f_type-eccentricity.png' %rsq_threshold)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['ecc'], recache=True,with_colorbar=True,with_curvature=True)

# Save this flatmap
filename = os.path.join(flatmap_out,'flatmap_space-fsaverage_rsq-%0.2f_type-size.png' %rsq_threshold)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['size'], recache=True,with_colorbar=True,with_curvature=True)

# Save this flatmap
filename = os.path.join(flatmap_out,'flatmap_space-fsaverage_rsq-%0.2f_type-rsquared.png' %rsq_threshold)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['rsq'], recache=True,with_colorbar=True,with_curvature=True)




## SOMATOTOPY ##

z_threshold = analysis_params['z_threshold']

soma_path = os.path.join(analysis_params['soma_outdir'],'sub-{sj}'.format(sj=sj),'run-median')

## group different body areas
face_zscore = np.load(os.path.join(soma_path,'z_face_contrast.npy'))
upper_zscore = np.load(os.path.join(soma_path,'z_upper_limb_contrast.npy'))
lower_zscore = np.load(os.path.join(soma_path,'z_lower_limb_contrast.npy'))

# threshold them
data_threshed_face = zthresh(face_zscore,threshold=z_threshold,side='both')
data_threshed_upper = zthresh(upper_zscore,threshold=z_threshold,side='both')
data_threshed_lower = zthresh(lower_zscore,threshold=z_threshold,side='both')

# combine 3 body part maps, threshold values
combined_zvals = np.array((face_zscore,upper_zscore,lower_zscore))

soma_labels, soma_zval = winner_takes_all(combined_zvals,analysis_params['all_contrasts'],threshold=z_threshold,side='above')

## Right vs left
RLupper_zscore = np.load(os.path.join(soma_path,'z_right-left_hand_contrast.npy'))
RLlower_zscore = np.load(os.path.join(soma_path,'z_right-left_leg_contrast.npy'))

# threshold left vs right, to only show relevant vertex
data_threshed_RLhand=zthresh(RLupper_zscore,side='both')
data_threshed_RLleg=zthresh(RLlower_zscore,side='both')

# all fingers in hand combined
LHfing_zscore = [] # load and append each finger z score in left hand list
RHfing_zscore = [] # load and append each finger z score in right hand list


print('Loading data for all fingers and appending in list')

for i in range(len(analysis_params['all_contrasts']['upper_limb'])//2):
    
    Ldata = np.load(os.path.join(soma_path,'z_%s-all_lhand_contrast.npy' %(analysis_params['all_contrasts']['upper_limb'][i])))
    Rdata = np.load(os.path.join(soma_path,'z_%s-all_rhand_contrast.npy' %(analysis_params['all_contrasts']['upper_limb'][i+5])))
   
    LHfing_zscore.append(Ldata)  
    RHfing_zscore.append(Rdata)

LHfing_zscore = np.array(LHfing_zscore)
RHfing_zscore = np.array(RHfing_zscore)

LH_labels, LH_zval = winner_takes_all(LHfing_zscore,
                                      analysis_params['all_contrasts']['upper_limb'][:5],side='above')

RH_labels, RH_zval = winner_takes_all(RHfing_zscore,
                                      analysis_params['all_contrasts']['upper_limb'][5:],side='above')


# all individual face regions combined

allface_zscore = [] # load and append each face part z score in list

print('Loading data for each face part and appending in list')

for _,name in enumerate(analysis_params['all_contrasts']['face']):
    
    facedata = np.load(os.path.join(soma_path,'z_%s-other_face_areas_contrast.npy' %(name)))   
    allface_zscore.append(facedata)  

allface_zscore = np.array(allface_zscore)

# combine them all in same array, winner takes all
allface_labels, allface_zval = winner_takes_all(allface_zscore,
                                      analysis_params['all_contrasts']['face'],side='above')

# define ROI by using face map, 
# to plot face part z score maps in relevant areas
face_roiF = np.zeros(data_threshed_face.shape) # set at 0 whatever is outside thresh   

for i,zsc in enumerate(data_threshed_face):
    if zsc > 0: # ROI only accounts for positive z score areas
        face_roiF[i]=allface_zval[i]
    

## create flatmaps for different parameters and save png

# vertex for face vs all others
images['v_face'] = cortex.Vertex(data_threshed_face.T, 'fsaverage',
                           vmin=-5, vmax=5,
                           cmap='BuBkRd')

# vertex for upper limb vs all others
images['v_upper'] = cortex.Vertex(data_threshed_upper.T, 'fsaverage',
                           vmin=-5, vmax=5,
                           cmap='BuBkRd')

# vertex for lower limb vs all others
images['v_lower'] = cortex.Vertex(data_threshed_lower.T, 'fsaverage',
                           vmin=-5, vmax=5,
                           cmap='BuBkRd')

# all somas combined
images['v_combined'] = cortex.Vertex2D(soma_labels.T, soma_zval.T, 'fsaverage',
                           vmin=0, vmax=1,
                           vmin2=z_threshold, vmax2=5, cmap='autumn_alpha')#BROYG_2D')#'my_autumn')

# vertex for right vs left hand
images['rl_upper'] = cortex.Vertex(data_threshed_RLhand.T, 'fsaverage',
                           vmin=-5, vmax=5,
                           cmap='bwr')

# vertex for right vs left leg
images['rl_lower'] = cortex.Vertex(data_threshed_RLleg.T, 'fsaverage',
                           vmin=-5, vmax=5,
                           cmap='bwr')


# all fingers left hand combined ONLY in left hand region 
# (as defined by LvsR hand contrast values)
images['v_Lfingers'] = cortex.Vertex2D(LH_labels.T, LH_zval.T, 'fsaverage',
                           vmin=0, vmax=1,
                           vmin2=z_threshold, vmax2=5, cmap='BROYG_2D')#BROYG_2D')#'my_autumn')

# all fingers right hand combined ONLY in right hand region 
# (as defined by LvsR hand contrast values)
images['v_Rfingers'] = cortex.Vertex2D(RH_labels.T, RH_zval.T, 'fsaverage',
                           vmin=0, vmax=1,
                           vmin2=z_threshold, vmax2=5, cmap='BROYG_2D')#BROYG_2D')#'my_autumn')

# 'eyebrows', 'eyes', 'tongue', 'mouth', combined
images['v_facecombined'] = cortex.Vertex2D(allface_labels.T, allface_zval.T, 'fsaverage',
                           vmin=0, vmax=1,
                           vmin2=z_threshold, vmax2=5, cmap='BROYG4col_2D') #costum colormap added to database

images['v_facecombinedROIF'] = cortex.Vertex2D(allface_labels.T, face_roiF.T, 'fsaverage',
                           vmin=0, vmax=1,
                           vmin2=z_threshold, vmax2=5, cmap='BROYG4col_2D')#BROYG_2D')#'my_autumn')


# Save this flatmap
filename = os.path.join(flatmap_out,'flatmap_space-fsaverage_zthresh-%0.2f_type-faceVSall.png' %z_threshold)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_face'], recache=True,with_colorbar=True,with_curvature=True)

# Save this flatmap
filename = os.path.join(flatmap_out,'flatmap_space-fsaverage_zthresh-%0.2f_type-upperVSall.png' %z_threshold)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_upper'], recache=True,with_colorbar=True,with_curvature=True)

# Save this flatmap
filename = os.path.join(flatmap_out,'flatmap_space-fsaverage_zthresh-%0.2f_type-lowerVSall.png' %z_threshold)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_lower'], recache=True,with_colorbar=True,with_curvature=True)

# Save this flatmap
filename = os.path.join(flatmap_out,'flatmap_space-fsaverage_zthresh-%0.2f_type-FULcombined.png' %z_threshold)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_combined'], recache=True,with_colorbar=True,with_curvature=True)

# Save this flatmap
filename = os.path.join(flatmap_out,'flatmap_space-fsaverage_zthresh-%0.2f_type-rightVSleftHAND.png' %z_threshold)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['rl_upper'], recache=True,with_colorbar=True,with_curvature=True)

# Save this flatmap
filename = os.path.join(flatmap_out,'flatmap_space-fsaverage_zthresh-%0.2f_type-rightVSleftLEG.png' %z_threshold)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['rl_lower'], recache=True,with_colorbar=True,with_curvature=True)

# Save this flatmap
filename = os.path.join(flatmap_out,'flatmap_space-fsaverage_zthresh-%0.2f_type-RHfing.png' %z_threshold)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_Rfingers'], recache=True,with_colorbar=True,with_curvature=True)

# Save this flatmap
filename = os.path.join(flatmap_out,'flatmap_space-fsaverage_zthresh-%0.2f_type-LHfing.png' %z_threshold)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_Lfingers'], recache=True,with_colorbar=True,with_curvature=True)

# Save this flatmap
filename = os.path.join(flatmap_out,'flatmap_space-fsaverage_zthresh-%0.2f_type-facecombined.png' %z_threshold)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_facecombined'], recache=True,with_colorbar=True,with_curvature=True)

# Save this flatmap
filename = os.path.join(flatmap_out,'flatmap_space-fsaverage_zthresh-%0.2f_type-facecombined_ROI-faceVsall.png' %z_threshold)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_facecombinedROIF'], recache=True,with_colorbar=True,with_curvature=True)


ds = cortex.Dataset(**images)
#cortex.webshow(ds, recache=True)
# Creates a static webGL MRI viewer in your filesystem
#web_path =os.path.join(analysis_params['cortex_dir'],'sub-{sj}'.format(sj=sj))
#cortex.webgl.make_static(outpath=web_path, data=ds) #, recache=True)#,template = 'cortex.html'){'polar':vrgba,'ecc':vecc}


