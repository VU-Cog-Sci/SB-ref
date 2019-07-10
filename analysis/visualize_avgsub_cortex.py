



import os, json
import sys, glob
import numpy as np
import cortex
import matplotlib.colors as colors
from utils import *
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as matcm
import matplotlib.pyplot as plt


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
print('averaging %d subjects' %(len(allsubs)))

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
combined_zvals = np.array((data_threshed_face,data_threshed_upper,data_threshed_lower))

print('Computing center of mass for different regions combined')
soma_labels, soma_zval = zsc_2_COM(combined_zvals)
#soma_labels, soma_zval = winner_takes_all(combined_zvals,analysis_params['all_contrasts'],threshold=z_threshold,side='above')

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

# compute center of mass and appropriate z-scores for each hand
print('Computing center of mass for left hand fingers')
LH_COM , LH_avgzval = zsc_2_COM(LHfing_zscore)
print('Computing center of mass for right hand fingers')
RH_COM , RH_avgzval = zsc_2_COM(RHfing_zscore)


# all individual face regions combined

allface_zscore = [] # load and append each face part z score in list

print('Loading data for each face part and appending in list')

for _,name in enumerate(analysis_params['all_contrasts']['face']):
    
    facedata = np.load(os.path.join(soma_path,'z_%s-other_face_areas_contrast.npy' %(name)))   
    allface_zscore.append(facedata)  

allface_zscore = np.array(allface_zscore)

# combine them all in same array

print('Computing center of mass for face elements %s' %(analysis_params['all_contrasts']['face']))
allface_COM , allface_avgzval = zsc_2_COM(allface_zscore)


## make colormap for elements ########

# same as the alpha colormap, then I can save both
my_colors = [(0, 0, 1),
          (0.27451,  0.94118 , 0.94118),
          (0, 1, 0),
          (1, 1, 0),
          #(1, 0.5, 0),
          (1, 0, 0)
          #(0.56078,0,1),
          #(0.29412,  0, 0.50980)
            ]  # B -> G -> Y -> #O-> R -> #L -> #P
n_bins = 100  # Discretizes the interpolation into bins
cmap_name = 'my_rainbow'
my_cm = LinearSegmentedColormap.from_list(cmap_name, my_colors, N=n_bins)
matcm.register_cmap(name=cmap_name, cmap=my_cm) # register it in matplotlib lib

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

create_my_colormaps(mapname='mycolormap_HSV_alpha.png')

## create flatmaps for different parameters and save png

# vertex for face vs all others
images['v_face'] = cortex.Vertex(data_threshed_face.T, 'fsaverage',
                           vmin=-7, vmax=7,
                           cmap='BuBkRd')

# vertex for upper limb vs all others
images['v_upper'] = cortex.Vertex(data_threshed_upper.T, 'fsaverage',
                           vmin=-7, vmax=7,
                           cmap='BuBkRd')

# vertex for lower limb vs all others
images['v_lower'] = cortex.Vertex(data_threshed_lower.T, 'fsaverage',
                           vmin=-7, vmax=7,
                           cmap='BuBkRd')

# all somas combined
images['v_combined'] = cortex.Vertex(soma_labels.T, 'fsaverage',
                           vmin=0, vmax=2,
                           cmap='autumn')#'J4') #costum colormap added to database

images['v_combined_alpha'] = cortex.Vertex2D(soma_labels.T, soma_zval.T, 'fsaverage',
                           vmin=0, vmax=2,
                           vmin2=min(soma_zval), vmax2=7, cmap='autumn_alpha')#BROYG_2D')#'my_autumn')

# vertex for right vs left hand
images['rl_upper'] = cortex.Vertex(data_threshed_RLhand.T, 'fsaverage',
                           vmin=-5, vmax=5,
                           cmap=blue_red)

# vertex for right vs left leg
images['rl_lower'] = cortex.Vertex(data_threshed_RLleg.T, 'fsaverage',
                           vmin=-5, vmax=5,
                           cmap=blue_red)

# all fingers left hand combined ONLY in left hand region 
# (as defined by LvsR hand contrast values)
images['v_Lfingers'] = cortex.Vertex(LH_COM.T, 'fsaverage',
                           vmin=0, vmax=4,
                           cmap=my_cm)#'J4')#costum colormap added to database

# all fingers right hand combined ONLY in right hand region 
# (as defined by LvsR hand contrast values)
images['v_Rfingers'] = cortex.Vertex(RH_COM.T, 'fsaverage',
                           vmin=0, vmax=4,
                           cmap=my_cm)#'J4')#costum colormap added to database

# 'eyebrows', 'eyes', 'tongue', 'mouth', combined
images['v_facecombined'] = cortex.Vertex(allface_COM.T, 'fsaverage',
                           vmin=0, vmax=3,
                           cmap=my_cm)#'J4') #costum colormap added to database

# same flatmaps but with alpha val related to z-scores
images['v_Lfingers_alpha'] = cortex.Vertex2D(LH_COM.T,LH_avgzval.T, 'fsaverage',
                           vmin=0, vmax=4,
                           vmin2=0, vmax2=2,
                           cmap='mycolormap_HSV_alpha')

images['v_Rfingers_alpha'] = cortex.Vertex2D(RH_COM.T,RH_avgzval.T, 'fsaverage',
                           vmin=0, vmax=4,
                           vmin2=0, vmax2=2,
                           cmap='mycolormap_HSV_alpha')

images['v_facecombined_alpha'] = cortex.Vertex2D(allface_COM.T,allface_avgzval.T, 'fsaverage',
                           vmin=0, vmax=3,
                           vmin2=0, vmax2=2,
                           cmap='mycolormap_HSV_alpha')

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
filename = os.path.join(flatmap_out,'flatmap_space-fsaverage_zthresh-%0.2f_type-FULcombined_alpha.png' %z_threshold)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_combined_alpha'], recache=True,with_colorbar=True,with_curvature=True)

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
filename = os.path.join(flatmap_out,'flatmap_space-fsaverage_zthresh-%0.2f_type-RHfing_alpha.png' %z_threshold)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_Rfingers_alpha'], recache=True,with_colorbar=True,with_curvature=True)

# Save this flatmap
filename = os.path.join(flatmap_out,'flatmap_space-fsaverage_zthresh-%0.2f_type-LHfing_alpha.png' %z_threshold)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_Lfingers_alpha'], recache=True,with_colorbar=True,with_curvature=True)

# Save this flatmap
filename = os.path.join(flatmap_out,'flatmap_space-fsaverage_zthresh-%0.2f_type-facecombined_alpha.png' %z_threshold)
print('saving %s' %filename)
_ = cortex.quickflat.make_png(filename, images['v_facecombined_alpha'], recache=True,with_colorbar=True,with_curvature=True)


ds = cortex.Dataset(**images)
#cortex.webshow(ds, recache=True)
# Creates a static webGL MRI viewer in your filesystem
#web_path =os.path.join(analysis_params['cortex_dir'],'sub-{sj}'.format(sj=sj))
#cortex.webgl.make_static(outpath=web_path, data=ds) #, recache=True)#,template = 'cortex.html'){'polar':vrgba,'ecc':vecc}


