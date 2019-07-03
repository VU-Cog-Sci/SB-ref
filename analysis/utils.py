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

from PIL import Image


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
        image = Image.open(png).convert('RGB')
        image = image.resize((screen[0],screen[1]), Image.ANTIALIAS) 
        
        im_gr_resc[i,:,:] = rescale(color.rgb2gray(np.asarray(image)), scale)
    
    img_bin = np.zeros(im_gr_resc.shape) #binary image, according to triangle threshold
    for i in range(len(im_gr_resc)):
        img_bin[i,:,:] = cv2.threshold(im_gr_resc[i,:,:],threshold_triangle(im_gr_resc[i,:,:]),255,cv2.THRESH_BINARY_INV)[1]
    
    # save as numpy array
    np.save(outfile, img_bin.astype(np.uint8))
    

def highpass_gii(filenames,polyorder,deriv,window,outpth):
    
    
    filenames_sg = []
    filenames.sort() #to make sure their in right order
    
    for run in range(4):#filenames)//2): # for every run (2 hemi per run)

        run_files = [x for _,x in enumerate(filenames) if 'run-0'+str(run+1) in os.path.split(x)[-1]]
        
        if not run_files:
            print('no soma files for run-0%s' %str(run+1))
        else:
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

  
def zthresh(zfile_in,threshold=0,side='above'):

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


def winner_takes_all(zfiles,labels,threshold=0,side='above'):
    
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
    

def mask_data(data,zscores,threshold=0,side='above'):
    ##################################################
    #    inputs:
    #        data1 - "original" data array (t,vertex)
    #        zscores - ROI zscore map, used to mask data1 (vertex,)
    #        threshold - value to threshold the zscores
    #        side - 'above'/'below'/'both', indicating if output values will be 
    #               above mean (positive zscores), below mean (negative zscores) or both
    #    outputs:
    #        maskdata - data array, masked 
    ##################################################
    maskdata = data.copy()
    
    for pos,vxl in enumerate(zscores):

        if side == 'above':
            if vxl < threshold or np.isnan(vxl):
                maskdata[:,pos]=np.nan 
        elif side == 'below':
            if vxl > -threshold or np.isnan(vxl):
                maskdata[:,pos]=np.nan 
        elif side == 'both':
            if vxl > -threshold or vxl < threshold or np.isnan(vxl):
                maskdata[:,pos]=np.nan 
    
    return maskdata


def make_contrast(dm_col,tasks,contrast_val=[1],num_cond=1):
    ##################################################
    #    inputs:
    #        dm_col - design matrix columns (all possible task names in list)
    #        tasks - list with list of tasks to give contrast value
    #                if num_cond=1 : [tasks]
    #                if num_cond=2 : [tasks1,tasks2], contrast will be tasks1 - tasks2 
    #        contrast_val - list with values for contrast
    #                if num_cond=1 : [value]
    #                if num_cond=2 : [value1,value2], contrast will be tasks1 - tasks2 
    #        num_cond - if one task vs the rest (1), or if comparing 2 tasks (2)
    #    outputs:
    #        contrast - contrast array
    ##################################################
    
    contrast = np.zeros(len(dm_col))

    if num_cond == 1: # if only one contrast value to give ("task vs rest")
        
        for j,name in enumerate(tasks[0]):
            for i in range(len(contrast)):
                if dm_col[i] == name:
                    contrast[i] = contrast_val[0]    
                    
    elif num_cond == 2: # if comparing 2 conditions (task1 - task2)
        
        for k,lbl in enumerate(tasks):
            idx = []
            for i,val in enumerate(lbl):
                idx.extend(np.where([1 if val == label else 0 for _,label in enumerate(dm_col)])[0])

            val = contrast_val[0] if k==0 else contrast_val[1] # value to give contrast

            for j in range(len(idx)):
                for i in range(len(dm_col)):
                    if i==idx[j]:
                        contrast[i]=val
       
    print('contrast for %s is %s'%(tasks,contrast))
    return contrast

def leave_one_out_lists(input_list):
    ##################################################
    #    inputs:
    #        input_list - list of item
    #
    #    outputs:
    #        out_lists - list of lists, with each element
    #                  of the input_list left out of the returned lists once, in order.
    ##################################################

    out_lists = []
    for x in input_list:
        out_lists.append([y for y in input_list if y != x])

    return out_lists

