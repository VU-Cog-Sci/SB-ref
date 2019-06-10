#useful functions to use in other scripts

import re
import nibabel as nb
import numpy as np
import os

import imageio
from skimage import color
import cv2
from skimage.transform import rescale
from skimage.filters import threshold_triangle

from nilearn import surface
from scipy.signal import savgol_filter

import pandas as pd
from spynoza.filtering.nodes import savgol_filter_confounds
from sklearn.decomposition import PCA



def median_gii(files,outdir):
    
    ##################################################
    #    inputs:
    #        files - list of absolute filenames to do median over
    #        outdir - path to save new files
    #    outputs:
    #        median_file - absolute output filename
    ##################################################
    
    
    img = []
    for i,filename in enumerate(files):
        img_load = nb.load(filename)
        img.append([x.data for x in img_load.darrays]) #(runs,TRs,vertices)
    
    median_img = np.median(img,axis=0)
    
    darrays = [nb.gifti.gifti.GiftiDataArray(d) for d in median_img]
    median_gii = nb.gifti.gifti.GiftiImage(header=img_load.header,
                                           extra=img_load.extra,
                                           darrays=darrays) # need to save as gii again

    median_file = os.path.join(outdir,re.sub('run-\d{2}_','run-median_',os.path.split(files[0])[-1]))
    nb.save(median_gii,median_file)

    return median_file



def screenshot2DM(filenames,scale,screen,outfile):
    
    ##################################################
    #    inputs:
    #        filenames - list of absolute filenames of pngs
    #        scale - scaling factor, to downsample images
    #        screen - list of screen resolution [hRes,vRes]
    #        outdir - path to save new files
    #    outputs:
    #        DM - absolute output design matrix filename
    ##################################################
    
    
    im_gr_resc = np.zeros((len(filenames),int(screen[1]*scale),int(screen[0]*scale)))
    
    for i, png in enumerate(filenames): #rescaled and grayscaled images
        im_gr_resc[i,:,:] = rescale(color.rgb2gray(imageio.imread(png)), scale)
    
    img_bin = np.zeros(im_gr_resc.shape) #binary image, according to triangle threshold
    for i in range(len(im_gr_resc)):
        img_bin[i,:,:] = cv2.threshold(im_gr_resc[i,:,:],threshold_triangle(im_gr_resc[i,:,:]),255,cv2.THRESH_BINARY_INV)[1]
    
    # save as numpy array
    np.save(outfile, img_bin.astype(np.uint8))


def highpass_gii(filenames,polyorder,deriv,window,outpth):
    
    
    filenames_sg = []
    filenames.sort() #to make sure their in right order
    
    for run in range(len(filenames)//2): # for every run (2 hemi per run)

        run_files = [x for _,x in enumerate(filenames) if 'run-0'+str(run+1) in os.path.split(x)[-1]]

        data_both = []
        for _,hemi in enumerate(run_files):
            data_both.append(surface.load_surf_data(hemi).T) #load surface data

        data_both = np.hstack(data_both) #stack then filter
        print('filtering run %s' %run_files)
        data_both -= savgol_filter(data_both, window, polyorder, axis=0,deriv=deriv)
        
        filenames_sg.append(data_both)
    
    return np.array(filenames_sg)

def highpass_confounds(confounds,nuisances,polyorder,deriv,window,tr,outpth):
    

    all_confs = []
    filt_conf_dir = []
    confounds.sort()

    for _,val in enumerate(confounds):
        # high pass confounds
        confounds_SG = savgol_filter_confounds(val, polyorder=polyorder, deriv=deriv, window_length=window, tr=tr)

        confs = pd.read_csv(confounds_SG, sep='\t', na_values='n/a')
        confs = confs[nuisances]

        #choose the minimum number of principal components such that at least 95% of the variance is retained.
        #pca = PCA(0.95,whiten=True) 
        pca = PCA(n_components=2,whiten=True) #had to chose 2 because above formula messes up len of regressors 
        pca_confs = pca.fit_transform(np.nan_to_num(confs))
        print('%d components selected for run' %pca.n_components_)

        # make list of dataframes 
        all_confs.append(pd.DataFrame(pca_confs, columns=['comp_{n}'.format(n=n) for n in range(pca.n_components_)]))

        # move file to median directory
        outfile = os.path.join(outpth,os.path.basename(confounds_SG))
        print('filtered confounds saved in %s' %outfile)

        filt_conf_dir.append(outfile)                      
        os.rename(confounds_SG, outfile)

    return filt_conf_dir

  
def zthresh(zfile_in,threshold,side='above'):

##################################################
#    inputs:
#        zfile_in - array with z scores
#        threshold - value to threshold the zscores
#        side - 'above'/'below'/'both', indicating if output values will be 
#               above mean (positive zscores), below mean (negative zscores) or both
#    outputs:
#        zfile_out - array with threshed z scores
##################################################

data_threshed = np.zeros(zfile_in.shape) # set at 0 whatever is outside thresh

for i,value in enumerate(zfile_in):
    if side == 'above':
        if value > threshold:
            data_threshed[i]=value
    elif side == 'below':
        if value < -threshold:
            data_threshed[i]=value
    elif side == 'both':
        if value < -threshold or value > threshold:
            data_threshed[i]=value

zfile_out = data_threshed

return zfile_out  


def winner_takes_all(zfiles,labels,threshold,side='above'):
    
    ##################################################
    #    inputs:
    #        zfiles - numpy array of zfiles for each condition
    #        labels - dictionary of labels to give to each condition
    #        threshold - value to threshold the zscores
    #        side - 'above'/'below'/'both', indicating if output values will be 
    #               above mean (positive zscores), below mean (negative zscores) or both
    #    outputs:
    #        all_zval - array with threshed z scores
    #        all_labels - array with corresponding labels
    ##################################################
        
    all_labels = np.zeros(zfiles[0].shape)
    all_zval = np.zeros(zfiles[0].shape)
    
    lbl = np.linspace(0,1,num=len(labels), endpoint=True)
    
    for i in range(len(all_labels)):
        if side == 'above': #only save values above mean
            
            zvals = [file[i] for _,file in enumerate(zfiles)] #zscore for each condition in that vertex
            max_zvals = max(zvals) #choose max one
    
            if max_zvals > threshold: #if bigger than thresh
                all_zval[i] = max_zvals #take max value for position, that will be the label shown
            
                for j,val in enumerate(lbl):
                    if np.argmax(zvals) == j: #if max zscore index = index of label
                        all_labels[i] = val #give that label
    
    return all_labels, all_zval
    



