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

from scipy.misc import imsave
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from nilearn.signal import clean

import cortex

from scipy import ndimage
from scipy import signal

import time


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
    

def highpass_gii(filenames,polyorder,deriv,window,outpth,combine_hemi=False):

    ##################################################
    #    inputs:
    #        filenames - list of absolute filenames for gii files (L and R hemi)
    #        polyorder - order of the polynomial used to fit the samples - must be less than window_length.
    #        deriv - order of the derivative to compute - must be a nonnegative integer
    #        window -  length of the filter window (number of coefficients) - must be a positive odd integer
    #        outpth - path to save new files
    #    outputs:
    #        filenames_sg - np array with all filtered runs appended
    #        filepaths_sg - list with filenames
    ##################################################
    
    
    filenames_sg = []
    filepaths_sg = []
    filenames.sort() #to make sure their in right order

    all_runs = np.arange(1,11) #10 is max number of runs for any of the tasks (FN is the biggest one)

    for run in all_runs: # for every run (2 hemi per run)

        run_files = [x for _,x in enumerate(filenames) if 'run-'+str(run).zfill(2) in os.path.split(x)[-1]]
        
        if not run_files:
            print('no files for run-%s' %str(run).zfill(2))
        else:

            if combine_hemi==False: # if we dont want to combine the hemi fields in one array

                for _,hemi in enumerate(run_files):

                    data_hemi = surface.load_surf_data(hemi).T #load surface data
                    print('filtering run %s' %hemi)
                    data_hemi -= savgol_filter(data_hemi, window, polyorder, axis=0,deriv=deriv)

                    name = os.path.splitext(os.path.splitext(os.path.split(hemi)[-1])[0])[0]
                    name = name+'_sg'

                    output = os.path.join(outpth,name)

                    np.save(output,data_hemi)

                    filenames_sg.append(data_hemi)
                    filepaths_sg.append(output+'.npy')

            else:

                data_both = []
                for _,hemi in enumerate(run_files):
                    data_both.append(surface.load_surf_data(hemi).T) #load surface data

                data_both = np.hstack(data_both) #stack then filter
                print('filtering run %s' %run_files)
                data_both -= savgol_filter(data_both, window, polyorder, axis=0,deriv=deriv)

                name = os.path.splitext(os.path.splitext(os.path.split(run_files[0])[-1])[0])[0]
                name = name.replace('hemi-L','hemi-both_sg')
                output = os.path.join(outpth,name)

                np.save(output,data_both)

                filenames_sg.append(data_both)
                filepaths_sg.append(output+'.npy')

    return np.array(filenames_sg),filepaths_sg
    

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

    data_threshed = np.zeros(zfile_in.shape);data_threshed[:]=np.nan # set at nan whatever is outside thresh

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


def zsc_2_COM(zdata):
    
##################################################
#    inputs:
#        zdata - array with z scores (elements,vertices)
#    outputs:
#        center_of_mass - array with COM for each vertex
#        avg_zval - array with average z-scores for each vertex
##################################################
    center_of_mass = []
    avg_zval = []
    for vrtx in range(zdata.shape[1]):
        elemz = zdata[...,vrtx] # save z-scores for all elements (ex:5 fing) of 1 vertex in array

        elemz_thresh = np.zeros(elemz.shape) # set to 0 negative z-scores, to ensure COM within element range
        f_zval = []
        for f,fval in enumerate(elemz):
            if fval > 0:
                elemz_thresh[f]=fval
                f_zval.append(fval)

        elem_num = np.linspace(0,zdata.shape[0]-1,num=zdata.shape[0])
        center_of_mass.append(sum(np.multiply(elem_num,elemz_thresh))/sum(elemz_thresh))
        avg_zval.append(np.average(f_zval))

    center_of_mass = np.array(center_of_mass)
    avg_zval = np.array(avg_zval)

    
    return center_of_mass,avg_zval

    
def create_my_colormaps(mapname='mycolormap_HSV_alpha.png'):
   
    hue, alpha = np.meshgrid(np.linspace(
        0.7,0, 80, endpoint=False), 1-np.linspace(0, 1, 80)) #values chosen to make it visible
    print(hue.shape)
    hsv = np.zeros(list(hue.shape)+[3])
    print(hsv.shape)
    # convert angles to colors, using correlations as weights
    hsv[..., 0] = hue  # angs_discrete  # angs_n
    # np.sqrt(rsq) #np.ones_like(rsq)  # np.sqrt(rsq)
    hsv[..., 1] = np.ones_like(alpha)
    # np.nan_to_num(rsq ** -3) # np.ones_like(rsq)#n
    hsv[..., 2] = np.ones_like(alpha)

    rgb = colors.hsv_to_rgb(hsv)
    rgba = np.vstack((rgb.T, alpha[..., np.newaxis].T)).T
    #plt.imshow(rgba)
    hsv_fn = os.path.join(os.path.split(cortex.database.default_filestore)[
                          0], 'colormaps', mapname)
    imsave(hsv_fn, rgba)
        

def clean_confounds(npdata,confounds,outpth,combine_hemi=False):
    
    ##################################################
    #    inputs:
    #        npdata - list of absolute filenames for numpy array data
    #        confounds - list of absolute filenames for confound tsvs
    #        outpth - path to save new files
    #    outputs:
    #        new_data - np array with all filtered runs appended
    #        new_data_pth - list with absolute filenames
    ##################################################
    
    #sort to make sure lists in right order
    npdata.sort()
    confounds.sort()
    new_data = []
    new_data_pth = []
    counter = 0


    for idx,file in enumerate(npdata):
        print('regressing out confounds from %s' %(file))
        data =np.load(file) #load data for run

        confs = pd.read_csv(confounds[counter], sep='\t', na_values='n/a') #load tsv

        # if even index, and both hemifields in list increase counter, or if hemi already combined
        if idx == 0:
            counter = 0
        elif (combine_hemi == False and idx % 2 == 0) or (combine_hemi == True): 
            counter += 1

        d = clean(data, confounds=confs.values, standardize=False) #clean it
        
        name = os.path.splitext(os.path.split(file)[-1])[0]+'_conf'
        output = os.path.join(outpth,name)
        
        np.save(output,d)
        print('clean data saved in %s' %(output))
        new_data.append(d)
        new_data_pth.append(output+'.npy')
    
    return new_data,new_data_pth

    
def nparray2mgz(nparray,giifiles,outdir):
    
    ##################################################
    #    inputs:
    #        nparray - list of absolute path for np arrays (all hemi and runs)
    #        giifiles - list of absolute path for gii files (needs to be analogous to above)
    #        outdir - output dir
    #    outputs:
    #        mgz_files - list of absolute path for files
    ##################################################
    
    # make sure in right order
    nparray.sort()
    giifiles.sort()
    mgz_files = []
    
    for index,file in enumerate(giifiles):
        
        gii_load = nb.load(file) #load original hemi gii file
        nparr = np.load(nparray[index]) # load processed hemi np array
        
        darrays = [nb.gifti.gifti.GiftiDataArray(d) for d in nparr]
        
        # new gii file is the processed numpy array as gii
        new_gii = nb.gifti.gifti.GiftiImage(header=gii_load.header,
                                           extra=gii_load.extra,
                                           darrays=darrays)
        new_gii_pth = os.path.join(outdir,os.path.splitext(os.path.split(nparray[index])[-1])[0]+'.func.gii')
        nb.save(new_gii,new_gii_pth)
        print('saved numpy array as gifti in %s' %(new_gii_pth))
        
        new_mgz = os.path.join(outdir,os.path.splitext(os.path.split(nparray[index])[-1])[0]+'.mgz')
        mgz_files.append(new_mgz)
        print('converting gifti to mgz as %s' %(new_mgz))
        os.system('mri_convert %s %s'%(new_gii_pth,new_mgz))
    
    return mgz_files


def median_mgz(files,outdir):
    
    ##################################################
    #    inputs:
    #        files - list of absolute filenames to do median over
    #        outdir - path to save new files
    #    outputs:
    #        median_file - absolute output filename
    ##################################################
    
    
    img = []
    for i,filename in enumerate(files):
        img_load = surface.load_surf_data(filename).T
        img.append(img_load) #(runs,TRs,vertices)
    
    median_img = np.median(img,axis=0)

    median_file = os.path.join(outdir,re.sub('run-\d{2}_','run-median_',os.path.split(files[0])[-1]))+'.npy'
    median_file = re.sub('smooth5.mgz','smooth5',median_file)
    np.save(median_file,median_img)

    return median_file


def median_pRFestimates(subdir,with_smooth=True):
    
    ####################
    #    inputs
    # subdir - absolute path to all subject dir (where fits are)
    # with_smooth - boolean, use smooth data?
    #   outputs
    # estimates - dictionary with average estimated parameters
    
    allsubs = os.listdir(subdir)
    allsubs.sort()
    print('averaging %d subjects' %(len(allsubs)))
    
    sub_list = []
    rsq = []
    xx = []
    yy = []
    size = []
    baseline = []
    beta = []
    
    for idx,sub in enumerate(allsubs):
        
        if with_smooth==True: #if data smoothed
            sub_list.append(os.path.join(subdir,sub,'run-median','smooth'))
        else:
            sub_list.append(os.path.join(subdir,sub,'run-median'))
        
        estimates_list = [x for x in os.listdir(sub_list[idx]) if x.endswith('estimates.npz') ]
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
        
    
    estimates = {'subs':sub_list,'r2':med_rsq,'x':med_xx,'y':med_yy,
                 'size':med_size,'baseline':med_baseline,'betas':med_beta}
    
    return estimates


def psc_gii(gii_file,outpth, method='median'):
    
    ##################################################
    #    inputs:
    #        gii_file - list of absolute filenames for giis to perform percent signal change
    #        outpth - path to save new files
    #        method - median vs mean
    #    outputs:
    #        new_gii_pth - list with absolute filenames for saved giis
    ##################################################
    
    # gii data
    gii_file.sort()
    new_gii_pth = []
    
    for index,file in enumerate(gii_file):
    
        img_load = nb.load(file) #doing this to get header info etc
        data_in = np.array([x.data for x in img_load.darrays])

        if method == 'mean':
            data_m = np.mean(data_in,axis=0)
        elif method == 'median':
            data_m = np.median(data_in, axis=0)

        data_conv = 100.0 * (data_in - data_m)/np.abs(data_m)

        new_name =  os.path.split(file)[-1] 
        new_name = new_name.replace('.func.gii','_psc.func.gii')
        full_pth = os.path.join(outpth,new_name)

        
        darrays = [nb.gifti.gifti.GiftiDataArray(d) for d in data_conv]
        new_gii = nb.gifti.gifti.GiftiImage(header=img_load.header,
                                           extra=img_load.extra,
                                           darrays=darrays) # need to save as gii again
        print('saving %s' %full_pth)
        nb.save(new_gii,full_pth) #save in correct path
        
        new_gii_pth.append(full_pth)

    return new_gii_pth


def sacc2longDM(saccfile,gazeinfo,outfilename,smp_freq=1000,subsmp_freq=50,nrTR=103,TR=1.6,fig_sfactor=0.1,screen=[1920, 1080]):
    ##################################################
    #    inputs:
    #        saccfile - absolute path to numpy array with saccade info
    #        gazeinfo - absolute path to numpy array with gaze info
    #        smp_freq - original sample frequency of eyetracking data
    #        subsmp_freq - frequency to downsample the data
    #        nrTR - number of TRs of FN data
    #        TR - in seconds
    #        fig_sfactor - scaling factor for figure
    #        screen - screen res
    ##################################################
    
    sac_data = np.load(saccfile) # array of (3 x trial length), filled with sacc amplitude, x position and y position of vector       
    trial_info = np.load(gazeinfo)#,allow_pickle=True)
    
    print('loading saccade data for %s' %saccfile)

    # define relevant timings
    start_scan = int(trial_info['trial_phase_info'][0][0]-trial_info['trl_str_end'][0][0]) #start of scan? relative to begining of trial
    start_movie = int(trial_info['trial_phase_info'][0][1]-trial_info['trl_str_end'][0][0]) #beginning of movie relative to begining of trial
    end_movie = int(trial_info['trial_phase_info'][0][2]-trial_info['trl_str_end'][0][0]) #end of movie relative to begining of trial
    end_trial = int(trial_info['trl_str_end'][0][1] - trial_info['trl_str_end'][0][0])

    # save array with relevant saccade data from 1st TR to end of trial
    amp_start_scan = [amp for _,amp in enumerate(sac_data['amplitude'][start_scan::])]
    xpos_start_scan = [xpos for _,xpos in enumerate(sac_data['xpos'][start_scan::])]
    ypos_start_scan = [ypos for _,ypos in enumerate(sac_data['ypos'][start_scan::])]
    
    # now save resampled within number of TRs
    expt_timepoints_indices = np.arange(0, nrTR * subsmp_freq * TR)

    amp_sliced = amp_start_scan[0::int(smp_freq/subsmp_freq)].copy()
    amp_resampTR = amp_sliced[:len(expt_timepoints_indices)]

    xpos_sliced = xpos_start_scan[0::int(smp_freq/subsmp_freq)].copy()
    xpos_resampTR = xpos_sliced[:len(expt_timepoints_indices)]

    ypos_sliced = ypos_start_scan[0::int(smp_freq/subsmp_freq)].copy()
    ypos_resampTR = ypos_sliced[:len(expt_timepoints_indices)]
    
    checkpoint = 0 # checkpoint counter, for sanity
    start_timer = time.time() # also added timer

    for smp_idx,_ in enumerate(expt_timepoints_indices): #saves images during actual scanning period
        # do loop over all samples to get numpy array with "screenshots"

        # plotted figure is 10x smaller, so also have to rescale values to fit
        x_centered = (xpos_resampTR[smp_idx] + screen[0]/2.0)#*fig_sfactor
        y_centered = (ypos_resampTR[smp_idx] + screen[1]/2.0)#*fig_sfactor
        amp_pix = (amp_resampTR[smp_idx]/2)#*fig_sfactor #diameter will be the amplitude of saccade

        sac_endpoint = plt.Circle((x_centered, y_centered), radius = amp_pix, color='r',clip_on = False) #important to avoid clipping of circle
        # res is figsiz*dpi, thus dividing by 100
        fig, ax = plt.subplots(figsize=(screen[0]*fig_sfactor,screen[1]*fig_sfactor), dpi=1) # note we must use plt.subplots, not plt.subplot 
        ax.set_xlim((0, screen[0]))#*fig_sfactor))
        ax.set_ylim((0, screen[1]))#*fig_sfactor))
        ax.add_artist(sac_endpoint)
        plt.axis('off')
        fig.canvas.draw()

        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        img_gray = color.rgb2gray(np.asarray(img))
        try:
            img_threshbin = cv2.threshold(img_gray,threshold_triangle(img_gray),255,cv2.THRESH_BINARY_INV)[1]
        except:
            img_threshbin = (img_gray*0).astype(np.uint8) # to make it black when no saccade
            pass

        if smp_idx==0: #binary image all samples stacked
            img_bin = np.expand_dims(img_threshbin, axis=0)
        else:
            img_bin = np.concatenate((img_bin,np.expand_dims(img_threshbin, axis=0)),axis=0)

        plt.close()

        if smp_idx==checkpoint:
            print('%d / %d took %d seconds' %(checkpoint,len(expt_timepoints_indices),(time.time()-start_timer)))
            checkpoint += 1000

    # save as numpy array
    np.save(outfilename, img_bin.astype(np.uint8))
    print('saved %s' %outfilename)
    
    # save as gif too, for fun/as check
    imageio.mimwrite(outfilename.replace('.npy','.gif'), img_bin.astype(np.uint8) , 'GIF')
