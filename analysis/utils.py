#useful functions to use in other scripts

import re
import nibabel as nb
import numpy as np
import os, json
import glob

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

from PIL import Image, ImageOps

from scipy.misc import imsave
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from nilearn.signal import clean

import cortex

from scipy import ndimage
from scipy import signal

import time

from nilearn.datasets import fetch_surf_fsaverage
import nilearn.plotting as ni_plt

import nipype.interfaces.freesurfer as fs

import math
from scipy.stats import pearsonr, t, norm

with open('analysis_params.json','r') as json_file:
            analysis_params = json.load(json_file)


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


def screenshot2DM(filenames,scale,screen,outfile,dm_shape = 'rectangle'):

    ##################################################
    #    inputs:
    #        filenames - list of absolute filenames of pngs
    #        scale - scaling factor, to downsample images
    #        screen - list of screen resolution [hRes,vRes]
    #        outdir - path to save new files
    #    outputs:
    #        DM - absolute output design matrix filename
    ##################################################
    
    hRes = int(screen[0])
    vRes = int(screen[1])
    
    if dm_shape == 'square': # make square dm, using max screen dim
        dim1 = hRes
        dim2 = hRes
        
    else:
        dim1 = hRes
        dim2 = vRes
        
    im_gr_resc = np.zeros((len(filenames),int(dim2*scale),int(dim1*scale)))
    
    for i, png in enumerate(filenames): #rescaled and grayscaled images
        image = Image.open(png).convert('RGB')
        
        if dm_shape == 'square': # add padding (top and bottom borders)
            #padded_img = Image.new(image.mode, (hRes, hRes), (255, 255, 255))
            #padded_img.paste(image, (0, ((hRes - vRes) // 2)))
            padding = (0, (hRes - vRes)//2, 0, (hRes - vRes)-((hRes - vRes)//2))
            image = ImageOps.expand(image, padding, fill=(255, 255, 255))
            #plt.imshow(image)
            
        image = image.resize((dim1,dim2), Image.ANTIALIAS)
        im_gr_resc[i,:,:] = rescale(color.rgb2gray(np.asarray(image)), scale)
    
    img_bin = np.zeros(im_gr_resc.shape) #binary image, according to triangle threshold
    for i in range(len(im_gr_resc)):
        img_bin[i,:,:] = cv2.threshold(im_gr_resc[i,:,:],threshold_triangle(im_gr_resc[i,:,:]),255,cv2.THRESH_BINARY_INV)[1]

    # save as numpy array
    np.save(outfile, img_bin.astype(np.uint8))


def highpass_gii(filename,polyorder,deriv,window,outpth):

    ##################################################
    #    inputs:
    #        filename - list of absolute filename for gii file
    #        polyorder - order of the polynomial used to fit the samples - must be less than window_length.
    #        deriv - order of the derivative to compute - must be a nonnegative integer
    #        window -  length of the filter window (number of coefficients) - must be a positive odd integer
    #        outpth - path to save new files
    #    outputs:
    #        filename_sg - np array with filtered run
    #        filepath_sg - filename
    ##################################################

    filename_sg = []
    filepath_sg = []

    if not os.path.isfile(filename): # check if file exists
            print('no file found called %s' %filename)
    else:

        # load with nibabel instead to save outputs always as gii
        gii_in = nb.load(filename)
        data_in = np.array([gii_in.darrays[i].data for i in range(len(gii_in.darrays))]) #load surface data

        print('filtering run %s' %filename)
        data_in_filt = savgol_filter(data_in, window, polyorder, axis=0,deriv=deriv,mode='nearest')
        data_out = data_in - data_in_filt + data_in_filt.mean(axis=0) # add mean image back to avoid distribution around 0

        darrays = [nb.gifti.gifti.GiftiDataArray(d) for d in data_out]
        gii_out = nb.gifti.gifti.GiftiImage(header=gii_in.header, extra=gii_in.extra, darrays=darrays)

        output = os.path.join(outpth,os.path.split(filename)[-1].replace('.func.gii','_sg.func.gii'))

        nb.save(gii_out,output) # save as gii file

        filename_sg = data_out
        filepath_sg = output

    return np.array(filename_sg),filepath_sg



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
    imageio.imwrite(hsv_fn, rgba)


def clean_confounds(gii_file,confounds,outpth):

    ##################################################
    #    inputs:
    #        npdata - absolute filename for gii
    #        confounds - absolute filename for confound tsv
    #        outpth - path to save new files
    #    outputs:
    #        new_data - np array with all filtered runs appended
    #        new_data_pth - list with absolute filenames
    ##################################################

    out_data = []
    out_data_pth = []

    if not os.path.isfile(gii_file): # check if file exists
        print('no file found called %s' %gii_file)
    else:
        print('regressing out confounds from %s' %(gii_file))
        # load with nibabel instead to save outputs always as gii
        gii_in = nb.load(gii_file)
        data_in = np.array([gii_in.darrays[i].data for i in range(len(gii_in.darrays))]) #load surface data

        confs = pd.read_csv(confounds, sep='\t', na_values='n/a') #load tsv

        data_clean = clean(data_in, confounds=confs.values, standardize=False) #clean it

        darrays = [nb.gifti.gifti.GiftiDataArray(d) for d in data_clean]
        new_gii = nb.gifti.gifti.GiftiImage(header=gii_in.header,
                                           extra=gii_in.extra,
                                           darrays=darrays) # need to save as gii again

        name = os.path.split(gii_file)[-1].replace('.func.gii','_conf.func.gii')

        out_data = np.array(data_clean)
        out_data_pth = os.path.join(outpth,name)

        print('saving %s' %out_data_pth)
        nb.save(new_gii,out_data_pth) #save in correct path


    return out_data,out_data_pth


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

    np.save(outdir,median_img)

    return outdir



def psc_gii(gii_file,outpth, method='median'):

    ##################################################
    #    inputs:
    #        gii_file - absolute filename for gii
    #        outpth - path to save new files
    #        method - median vs mean
    #    outputs:
    #        psc_gii - np array with percent signal changed file
    #        psc_gii_pth - list with absolute filenames for saved giis
    ##################################################

    psc_gii = []
    psc_gii_pth = []

    if not os.path.isfile(gii_file): # check if file exists
            print('no file found called %s' %gii_file)
    else:

        # load with nibabel instead to save outputs always as gii
        img_load = nb.load(gii_file)
        data_in = np.array([img_load.darrays[i].data for i in range(len(img_load.darrays))]) #load surface data

        print('PSC run %s' %gii_file)

        if method == 'mean':
            data_m = np.mean(data_in,axis=0)
        elif method == 'median':
            data_m = np.median(data_in, axis=0)

        data_conv = 100.0 * (data_in - data_m)/data_m#np.abs(data_m)

        new_name =  os.path.split(gii_file)[-1].replace('.func.gii','_psc.func.gii') # file name

        darrays = [nb.gifti.gifti.GiftiDataArray(d) for d in data_conv]
        new_gii = nb.gifti.gifti.GiftiImage(header=img_load.header,
                                           extra=img_load.extra,
                                           darrays=darrays) # need to save as gii again
        psc_gii = np.array(data_conv)
        psc_gii_pth = os.path.join(outpth,new_name)

        print('saving %s' %psc_gii_pth)
        nb.save(new_gii,psc_gii_pth) #save in correct path



    return psc_gii,psc_gii_pth



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



def plot_tSNR(gii_in,hemi,outpth,mesh='fsaverage'):

    ##################################################
    #    inputs:
    #        gii_in - absolute filename for gii file
    #        hemi - string with name of hemifield to plot ('right' or 'left')
    #        mesh - string with name of mesh to load for plotting (default 'fsaverage')
    #        outpth - path to save plot
    ##################################################

    surfmesh = fetch_surf_fsaverage(mesh=mesh)
    hemi_data = surface.load_surf_data(gii_in).T

    out_name = os.path.split(gii_in)[-1].replace('.func.gii','_tSNR.png')

    if not os.path.exists(outpth): # check if path to save plot exists
        os.makedirs(outpth)  #if not, create it

    if hemi == 'left':
        ni_plt.plot_surf_stat_map(surfmesh['infl_left'], stat_map=np.median(hemi_data,axis=0)/np.std(hemi_data,axis=0),
                                    hemi='left', view='lateral', colorbar=True,
                                    bg_map=surfmesh['sulc_left'], bg_on_data=True,darkness=0.5,
                                    title='tSNR map')
    else:
        ni_plt.plot_surf_stat_map(surfmesh['infl_right'], stat_map=np.median(hemi_data,axis=0)/np.std(hemi_data,axis=0),
                                    hemi='right', view='lateral', colorbar=True,
                                    bg_map=surfmesh['sulc_right'], bg_on_data=True,darkness=0.5,
                                    title='tSNR map')
    plt.savefig(os.path.join(outpth,out_name), bbox_inches="tight")



def smooth_gii(gii_file,outdir,fwhm=5):

    ##################################################
    #    inputs:
    #        gii_file - absolute path for gii file
    #        outdir - output dir
    #       fwhm - width of the kernel, at half of the maximum of the height of the Gaussian
    #    outputs:
    #        smooth_gii - np array with smoothed file
    #        smooth_gii_pth - absolute path for smoothed file
    ##################################################
    smooth_gii = []
    smooth_gii_pth = []

    if not os.path.isfile(gii_file): # check if file exists
            print('no file found called %s' %gii_file)
    else:

        # load with nibabel instead to save outputs always as gii
        gii_in = nb.load(gii_file)
        data_in = np.array([gii_in.darrays[i].data for i in range(len(gii_in.darrays))]) #load surface data

        print('loading file %s' %gii_file)

        # first need to convert to mgz
        # will be saved in output dir
        new_mgz = os.path.join(outdir,os.path.split(gii_file)[-1].replace('.func.gii','.mgz'))

        print('converting gifti to mgz as %s' %(new_mgz))
        os.system('mri_convert %s %s'%(gii_file,new_mgz))

        # now smooth it
        smoother = fs.SurfaceSmooth()
        smoother.inputs.in_file = new_mgz
        smoother.inputs.subject_id = 'fsaverage'

        # define hemisphere
        smoother.inputs.hemi = 'lh' if '_hemi-L' in new_mgz else 'rh'
        print('smoothing %s' %smoother.inputs.hemi)
        smoother.inputs.fwhm = fwhm
        smoother.run() # doctest: +SKIP

        new_filename = os.path.split(new_mgz)[-1].replace('.mgz','_smooth%d.mgz'%(smoother.inputs.fwhm))
        smooth_mgz = os.path.join(outdir,new_filename)
        os.rename(os.path.join(os.getcwd(),new_filename), smooth_mgz) #move to correct dir

        # transform to gii again
        new_data = surface.load_surf_data(smooth_mgz).T

        smooth_gii = np.array(new_data)
        smooth_gii_pth = smooth_mgz.replace('.mgz','.func.gii')
        print('converting to %s' %smooth_gii_pth)
        os.system('mri_convert %s %s'%(smooth_mgz,smooth_gii_pth))

    return smooth_gii,smooth_gii_pth

def highpass_pca_confounds(confounds,nuisances,polyorder,deriv,window,tr,outpth):

    # high pass confounds
    confounds_SG = savgol_filter_confounds(confounds, polyorder=polyorder, deriv=deriv, window_length=window, tr=tr)

    confs = pd.read_csv(confounds_SG, sep='\t', na_values='n/a')
    confs = confs[nuisances]

    #choose the minimum number of principal components such that at least 95% of the variance is retained.
    #pca = PCA(0.95,whiten=True)
    pca = PCA(n_components=2,whiten=True) #had to chose 2 because above formula messes up len of regressors
    pca_confs = pca.fit_transform(np.nan_to_num(confs))
    print('%d components selected for run' %pca.n_components_)

    # make list of dataframes
    all_confs = pd.DataFrame(pca_confs, columns=['comp_{n}'.format(n=n) for n in range(pca.n_components_)])

    # move file to median directory
    outfile = os.path.join(outpth,os.path.basename(confounds_SG))
    os.rename(confounds_SG, outfile)

    # save PCA data frame
    pca_outfile = outfile.replace('_sg.tsv','_sg_pca.tsv')
    all_confs.to_csv(pca_outfile, sep='\t', index=False)
    print('filtered and PCA confounds saved in %s' %pca_outfile)

    return pca_outfile

def plot_soma_timecourse(sj,run,task,vertex,giidir,eventdir,outdir,plotcolors=['#ad2f42','#59a89f','#9066ba'],template='fsaverage',extension='sg_psc.func.gii'):

    ##################################################
    #    inputs:
    #        sj - subject number
    #        run - run number (can also be median)
    #        vertex - vertex number for file
    #        giidir - absolute path to func file
    #        eventdir - absolute path to event file
    ##################################################

    data_both=[]
    for hemi_label in ['hemi-L','hemi-R']:

        filestring = os.path.join(giidir,'sub-{sj}_ses-*_task-soma_run-{run}_space-{template}_{hemi}_{ext}'.format(sj=str(sj).zfill(2),
                                                                                            run=str(run).zfill(2),
                                                                                            template=template,
                                                                                            hemi=hemi_label,
                                                                                            ext=extension))
        absfile = glob.glob(filestring) #absolute filename

        if not absfile: #if list is empty
            if run=='median':

                # list with absolute files to make median over
                run_files = [os.path.join(giidir,file) for _,file in enumerate(os.listdir(giidir))
                            if 'sub-{sj}'.format(sj=str(sj).zfill(2)) in file and
                            '_space-{template}'.format(template=template) in file and
                            '_{hemi}'.format(hemi=hemi_label) in file and
                             '_{ext}'.format(ext=extension) in file]
                run_files.sort()

                #compute and save median run
                filename = median_gii(run_files,giidir)
                print('computed %s' %(filename))

                # load surface data from path and append both hemi in array
                data_both.append(surface.load_surf_data(filename).T)
                print('loading %s' %filename)
            else:
                print('%s doesn\'t exist' %(absfile))
        else:
            # load surface data from path and append both hemi in array
            data_both.append(surface.load_surf_data(absfile[0]).T)
            print('loading %s' %absfile[0])

    # stack them to get 2D array
    data_both = np.hstack(data_both)

    #load events
    # list of stimulus onsets
    if run == 'median':
        print('no median event file, making standard times')
        events_inTR = np.linspace(7.5,132,num=60)
    else:
        events = [ev for _,ev in enumerate(os.listdir(eventdir)) if 'sub-'+str(sj).zfill(2) in ev and 'run-'+str(run).zfill(2) in ev and ev.endswith('events.tsv')]
        events = events[0]
        print('loading %s'%events)

        events_pd = pd.read_csv(os.path.join(eventdir,events),sep = '\t')

        new_events = []
        for ev in events_pd.iterrows():
            row = ev[1]
            new_events.append([row['onset'],row['duration'],row['trial_type']])

        df = pd.DataFrame(new_events, columns=['onset','duration','trial_type'])  #make sure only relevant columns present

        # event onsets in TR instead of seconds
        events_inTR = (np.linspace(df['onset'][0],df['onset'][len(df['onset'])-1],num = len(df['onset'])))/analysis_params['TR']

    # plot the fig
    fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

    color = {'face':plotcolors[0],'hand':plotcolors[1],'leg':plotcolors[2]}

    for idx,name in enumerate(task):

        # timeseries to plot
        timeseries = data_both[...,vertex[idx]]
        plt.plot(range(len(timeseries)),timeseries, linestyle='-',c=color[name],label='%s'%task[idx],marker='.')

    counter = 0
    while counter < len(events_inTR):
        face_line = np.arange(events_inTR[0+counter],events_inTR[4+counter],0.05)
        hand_line = np.arange(events_inTR[4+counter],events_inTR[9+counter],0.05)
        if counter==50:
            leg_line = np.arange(events_inTR[9+counter],events_inTR[9+counter]+2.25/1.6,0.05)
        else:
            leg_line = np.arange(events_inTR[9+counter],events_inTR[10+counter],0.05)
        plt.plot(face_line,[-5]*len(face_line),marker='s',c=color['face'])
        plt.plot(hand_line,[-5]*len(hand_line),marker='s',c=color['hand'])
        plt.plot(leg_line,[-5]*len(leg_line),marker='s',c=color['leg'])
        counter += 10

    plt.xlabel('Time (TR)',fontsize=18)
    plt.ylabel('BOLD signal change (%)',fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(0,len(timeseries))
    plt.legend(task, fontsize=10)
    plt.show()

    fig.savefig(os.path.join(outdir,'soma_timeseries_sub-{sj}_run-{run}.svg'.format(sj=str(sj).zfill(2),run=str(run).zfill(2))), dpi=100)



def median_pRFestimates(subdir,with_smooth=True,exclude_subs=['sub-07'],model='css',iterative=True):

####################
#    inputs
# subdir - absolute path to all subject dir (where fits are)
# with_smooth - boolean, use smooth data?
# exclude_subs=['subjects to be excluded from list']
#   outputs
# estimates - dictionary with average estimated parameters

    # make list of subject folders
    allsubs = [folder for _,folder in enumerate(os.listdir(subdir)) if 'sub-' in folder]
    allsubs.sort()
    print('averaging %d/%d subjects' %((len(allsubs)-len(exclude_subs)),len(allsubs)))
    print('using estimates from %s fit'%model)

    rsq = []
    xx = []
    yy = []
    size = []
    baseline = []
    beta = []
    ns = [] # for css
    new_subs = []

    exclude_counter = 0
    exclude_subs.sort() #then in order

    #load estimates and append
    for _,sub in enumerate(allsubs):

        if sub in exclude_subs[exclude_counter]:
            print('skipping %s, not included in average'%exclude_subs[exclude_counter])
            if len(exclude_subs)>(exclude_counter+1): exclude_counter += 1 # if more subs in list, increment counter
        else:

            # load prf estimates
            if with_smooth==True:    
                median_path = os.path.join(subdir,'{sj}'.format(sj=sub),'run-median','smooth%d'%analysis_params['smooth_fwhm'])
            else:
                median_path = os.path.join(subdir,'{sj}'.format(sj=sub),'run-median')

            if iterative==True: # if selecting params from iterative fit
                estimates_list = [x for x in os.listdir(median_path) if 'iterative' in x and x.endswith(model+'_estimates.npz')]
            else: # only look at grid fit
                estimates_list = [x for x in os.listdir(median_path) if 'iterative' not in x and x.endswith(model+'_estimates.npz')]
            estimates_list.sort() #sort to make sure pRFs not flipped

            estimates = []
            for _,val in enumerate(estimates_list) :
                print('appending %s'%val)
                estimates.append(np.load(os.path.join(median_path, val))) #save both hemisphere estimates in same array


            xx.append(np.concatenate((estimates[0]['x'],estimates[1]['x'])))
            yy.append(np.concatenate((estimates[0]['y'],estimates[1]['y'])))
               
            size.append(np.concatenate((estimates[0]['size'],estimates[1]['size'])))
            
            beta.append(np.concatenate((estimates[0]['betas'],estimates[1]['betas'])))
            baseline.append(np.concatenate((estimates[0]['baseline'],estimates[1]['baseline'])))
            
            if model=='css': ns.append(np.concatenate((estimates[0]['ns'],estimates[1]['ns']))) # exponent of css

            rsq.append(np.concatenate((estimates[0]['r2'],estimates[1]['r2'])))

            new_subs.append(sub)

    # get median values for each parameter
    xx = np.nanmedian(np.array(xx),axis=0)   
    yy = np.nanmedian(np.array(yy),axis=0)   

    size = np.nanmedian(np.array(size),axis=0)   
    beta = np.nanmedian(np.array(beta),axis=0)   
    baseline = np.nanmedian(np.array(baseline),axis=0)  

    if model=='css': 
        ns = np.nanmedian(np.array(ns),axis=0) 
    else: #if gauss
        ns = np.ones(xx.shape) 

    rsq = np.nanmedian(np.array(rsq),axis=0)

    estimates = {'subs':new_subs,'exclude_subs':exclude_subs,'r2':rsq,'x':xx,'y':yy,
                 'size':size,'baseline':baseline,'betas':beta,'ns':ns}

    return estimates

def shift_DM(prf_dm):
    # Very clunky and non-generic function, but works.
    # should optimize eventually

    # initialize a new DM with zeros, same shape as initial DM
    avg_prf_dm = np.zeros(prf_dm.shape)

    vert_bar_updown = range(13,22) #[13-21]
    vert_bar_downup = range(73,82) #[73-81]
    hor_bar_rightleft = range(24,41) #[24-40]
    hor_bar_leftright = range(54,71) #[54-70]

    # set vertical axis limits, to not plot above or below that
    # use first and last TR from initial bar pass (vertical up->down)
    vert_min_pix = np.where(prf_dm[0,:,vert_bar_updown[0]]==255)[0][0] # minimum vertical pixel index, below that should be empty (because no more display)
    vert_max_pix = np.where(prf_dm[0,:,vert_bar_updown[-1]]==255)[0][-1] # maximum vertical pixel index, above that should be empty (because no more display)

    # first get median width (grossura) of vertical and horizontal bars at a TR where full bar on screen
    length_vert_bar = int(np.median([len(np.where(prf_dm[x,:,vert_bar_updown[2]]==255)[0]) for x in range(prf_dm[:,:,vert_bar_updown[2]].shape[0])]))
    length_hor_bar = int(np.median([len(np.where(prf_dm[:,x,hor_bar_rightleft[2]]==255)[0]) for x in range(prf_dm[:,:,hor_bar_rightleft[2]].shape[1])]))

    # amount of pixel indexs I should shift bar forward in time -> (TR2 - TR1)/2
    shift_increment = math.ceil((np.median([np.where(prf_dm[x,:,vert_bar_updown[1]]==255)[0][-1] for x in range(prf_dm[:,:,vert_bar_updown[1]].shape[0])]) - \
        np.median([np.where(prf_dm[x,:,vert_bar_updown[0]]==255)[0][-1] for x in range(prf_dm[:,:,vert_bar_updown[0]].shape[0])]))/2)


    for j in range(prf_dm.shape[-1]): # FOR ALL TRs (j 0-89)

        # FOR VERTICAL BAR PASSES
        if j in vert_bar_updown or j in vert_bar_downup: 

            # loop to fill pixels that belong to the new bar position at that TR
            for i in range(length_vert_bar):
                if j in vert_bar_downup: 

                    if j==vert_bar_downup[-1]:

                        # shift end position and fill screen horizontally to make new bar
                        avg_end_pos = np.where(prf_dm[0,:,j]==255)[0][-1]-shift_increment

                        if avg_end_pos-i>=vert_min_pix: # if bigger that min pix index, which means it's within screen
                            avg_prf_dm[:,avg_end_pos-i,j]=255

                    else:
                        # shift start position and fill screen horizontally to make new bar
                        avg_start_pos = np.where(prf_dm[0,:,j]==255)[0][0]-shift_increment

                        if avg_start_pos+i<=vert_max_pix: # if lower that max pix index, which means it's within screen
                            avg_prf_dm[:,avg_start_pos+i,j]=255

                elif j in vert_bar_updown: #or j==vert_bar_downup[-1]:

                    if j==vert_bar_updown[-1]:

                        # shift start position and fill screen horizontally to make new bar
                        avg_start_pos = np.where(prf_dm[0,:,j]==255)[0][0]+shift_increment

                        if avg_start_pos+i<=vert_max_pix: # if lower that max pix index, which means it's within screen
                            avg_prf_dm[:,avg_start_pos+i,j]=255

                    else:
                        # shift end position and fill screen horizontally to make new bar
                        avg_end_pos = np.where(prf_dm[0,:,j]==255)[0][-1]+shift_increment

                        if avg_end_pos-i>=vert_min_pix: # if bigger that min pix index, which means it's within screen
                            avg_prf_dm[:,avg_end_pos-i,j]=255

        # FOR HORIZONTAL BAR PASSES
        if j in hor_bar_rightleft or j in hor_bar_leftright: 

            # loop to fill pixels that belong to the new bar position at that TR
            for i in range(length_hor_bar):

                if j in hor_bar_rightleft:
                    if j in hor_bar_rightleft[-2:]: # last two TRs might already be in limit, so fill based on other bar side

                        # shift end position and fill screen horizontally to make new bar
                        avg_end_pos = np.where(prf_dm[:,vert_min_pix,j]==255)[0][-1]-shift_increment

                        if avg_end_pos-i>=0: # if bigger than 0 (min x index), which means it's within screen
                            avg_prf_dm[avg_end_pos-i,vert_min_pix:vert_max_pix,j]=255

                    else:
                        avg_start_pos = np.where(prf_dm[:,vert_min_pix,j]==255)[0][0]-shift_increment

                        if avg_start_pos+i<=prf_dm.shape[0]-1: # if lower than 168 (max x index), which means it's within screen
                            avg_prf_dm[avg_start_pos+i,vert_min_pix:vert_max_pix,j]=255

                elif j in hor_bar_leftright:
                    if j in hor_bar_leftright[-2:]: # last two TRs might already be in limit, so fill based on other bar side

                        avg_start_pos = np.where(prf_dm[:,vert_min_pix,j]==255)[0][0]+shift_increment

                        if avg_start_pos+i<=prf_dm.shape[0]-1: # if lower than 168 (max x index), which means it's within screen
                            avg_prf_dm[avg_start_pos+i,vert_min_pix:vert_max_pix,j]=255

                    else:                    
                        # shift end position and fill screen horizontally to make new bar
                        avg_end_pos = np.where(prf_dm[:,vert_min_pix,j]==255)[0][-1]+shift_increment

                        if avg_end_pos-i>=0: # if bigger than 0 (min x index), which means it's within screen
                            avg_prf_dm[avg_end_pos-i,vert_min_pix:vert_max_pix,j]=255

    return avg_prf_dm #(x,y,t)


def crop_gii(gii_path,num_TR,outpath):
    ##################################################
    #    inputs:
    #        gii_path - (str) absolute filename for gii file
    #        num_TR - (int) number of TRs to remove from beginning of file
    #        outpath - (str) path to save new file
    #    outputs:
    #        crop_gii_path - (str) absolute filename for cropped gii file
    ##################################################

    # load with nibabel instead to save outputs always as gii
    gii_in = nb.load(gii_path)
    data_in = np.array([gii_in.darrays[i].data for i in range(len(gii_in.darrays))]) #load surface data

    crop_data = data_in[num_TR:,:] # crop initial TRs
    print('original file with %d TRs, now cropped and has %d TRs' %(data_in.shape[0],crop_data.shape[0]))

    # save as gii again
    darrays = [nb.gifti.gifti.GiftiDataArray(d) for d in crop_data]
    gii_out = nb.gifti.gifti.GiftiImage(header=gii_in.header, extra=gii_in.extra, darrays=darrays)

    crop_gii_path = os.path.join(outpath,os.path.split(gii_path)[-1].replace('.func.gii','_cropped.func.gii'))

    nb.save(gii_out,crop_gii_path) # save as gii file
    print('new file saved in %s'%crop_gii_path)

    return crop_gii_path

# calculate degrees of visual angle per pixel, to use for screen boundaries when plotting/masking
def dva_per_pix(height_cm,distance_cm,vert_res_pix):
    ##################################################
    #    inputs:
    #        height_cm - screen height
    #        distance_cm - screen distance (save unit as height)
    #        vert_res_pix - vertical resolution of screen
    #    outputs:
    #        deg_per_px - degree (dva) per pixel
    ##################################################
    
    deg_per_px = math.degrees(math.atan2(0.5*height_cm,distance_cm))/(0.5*vert_res_pix)

    return deg_per_px


# make masking function

def mask_estimates(x,y,size,beta,baseline,rsq,vertical_lim_dva,horizontal_lim_dva,ns=np.array([1]),max_size=10):
    
    ### inputs ######
    # estimates (np.array)
    # vertical_lim_dva (float) - vertical limit of screen in degrees
    # horizontal_lim_dva (float) - vertical limit of screen in degrees
    ## output ###
    # masked_estimates - dictionary with new arrays
    
    # make new variables that are masked 
    masked_xx = np.zeros(x.shape); masked_xx[:]=np.nan
    masked_yy = np.zeros(y.shape); masked_yy[:]=np.nan
    masked_size = np.zeros(size.shape); masked_size[:]=np.nan
    masked_beta = np.zeros(beta.shape); masked_beta[:]=np.nan
    masked_baseline = np.zeros(baseline.shape); masked_baseline[:]=np.nan
    masked_rsq = np.zeros(rsq.shape); masked_rsq[:]=np.nan
    masked_ns = np.zeros(ns.shape); masked_ns[:]=np.nan

    for i in range(len(x)): #for all vertices
        if x[i] <= horizontal_lim_dva and x[i] >= -horizontal_lim_dva: # if x within horizontal screen dim
            if y[i] <= vertical_lim_dva and y[i] >= -vertical_lim_dva: # if y within vertical screen dim
                if beta[i]>=0: # only account for positive RF
                    if size[i]<=max_size: # limit size to max size defined in fit

                        # save values
                        masked_xx[i]=x[i]
                        masked_yy[i]=y[i]
                        masked_size[i]=size[i]
                        masked_beta[i]=beta[i]
                        masked_baseline[i]=baseline[i]
                        masked_rsq[i]=rsq[i]
                        if len(ns)>1:
                            masked_ns[i]=ns[i]

    masked_estimates = {'x':masked_xx,'y':masked_yy,'size':masked_size,
                        'beta':masked_beta,'baseline':masked_baseline,'ns':masked_ns,
                        'rsq':masked_rsq}
    
    return masked_estimates


def fit_glm(voxel, dm):
    #### inputs ####
    #
    # voxel - single voxel timecourse
    # dm - design matrix (TRs,predictors)
    #
    #### outputs ####
    #
    # model - model fit for voxel
    # betas - betas for model
    # r2 - coefficient of determination
    # mse - mean of the squared residuals
    
    if np.isnan(voxel).any():
        betas = np.nan
        model = np.nan
        mse = np.nan
        r2 = np.nan
    else:   # if not nan (some vertices might have nan values)
        betas = np.linalg.lstsq(dm, voxel)[0]
        model = dm.dot(betas)

        mse = np.mean((model - voxel) ** 2) # calculate mean of squared residuals
        r2 = pearsonr(model, voxel)[0] ** 2 # and the rsq
    
    return model,betas,r2,mse
    

def compute_stats(voxel, dm, contrast,betas):
    #### inputs ####
    #
    # voxel - single voxel timecourse
    # dm - design matrix (TRs,predictors)
    # contrast - contrast vector
    # betas - beta values for voxel
    #
    #### outputs ####
    #
    # t_val - t-value
    # p_val - p-value
    # z_score 
    
    def design_variance(X, which_predictor=1):
        ''' Returns the design variance of a predictor (or contrast) in X.

        Parameters
        ----------
        X : numpy array
            Array of shape (N, P)
        which_predictor : int or list/array
            The index of the predictor you want the design var from
            (or contrast vector as a list/array).

        Returns
        -------
        des_var : float
            Design variance of the specified predictor from X.
        '''
    
        is_single = isinstance(which_predictor, int)
        if is_single:
            idx = which_predictor
        else:
            idx = np.array(which_predictor) != 0

        c = np.zeros(X.shape[1])
        c[idx] = 1 if is_single == 1 else which_predictor[idx]
        des_var = c.dot(np.linalg.pinv(X.T.dot(X))).dot(c.T)
        return des_var

    
    if np.isnan(voxel).any():
        t_val = np.nan
        p_val = np.nan
        z_score = np.nan

    else:   # if not nan (some vertices might have nan values)
        
        # calculate design variance
        design_var = design_variance(dm, contrast)
        
        # sum of squared errors
        sse = ((voxel - (dm.dot(betas))) ** 2).sum() 
        #degrees of freedom = timepoints - predictores
        df = (dm.shape[0] - dm.shape[1]) 
        
        # t statistic for vertex
        t_val = contrast.dot(betas) / np.sqrt((sse/df) * design_var)
        
        # p-value for voxel
        # t.sf() ALWAYS returns the right-tailed p-value
        # For negative t-values, however, you'd want the left-tailed p-value
        # hence passing the absolute t-value to the t.sf() function
        #p_val = t.sf(np.abs(t_val), (dm.shape[0] - dm.shape[1])) * 2
        
        #z_score = norm.ppf(p_val)
        
        # to ensure that zstats will not reach infinity, use this conversion:
        # (see http://www.stats.uwo.ca/faculty/aim/2010/JSSSnipets/V23N1.pdf)
        # This is because a p of .9999956 will be less precies than .0000044, 
        # as the latter is internally represented as 4.4e-6,
        # which leaves much more room for decimals.
        # To get pvals close to zero, make sure the t-stat is negative
        # and the cumulative distribution is taken up to that point
        p_val = 2*t.cdf(-np.abs(t_val), df = (dm.shape[0]-dm.shape[1]))
        ts = np.array(np.sign(t_val))
        z_score = -ts*norm.ppf(p_val)

    return t_val,p_val,z_score
    

def make_median_soma_sub(all_subs,file_extension,out_dir,median_gii=median_gii):
    # input sub list, file extension, out_dir, function to make median run

    for idx,sub in enumerate(all_subs):
        # path to functional files
        filepath = glob.glob(os.path.join(analysis_params['post_fmriprep_outdir'], 'soma', 'sub-{sj}'.format(sj=sub), '*'))
        print('functional files from %s' % os.path.split(filepath[0])[0])

        # list of functional files (5 runs)
        filename = [run for run in filepath if 'soma' in run and 'fsaverage' in run and run.endswith(file_extension)]
        filename.sort()

        ##### compute median run for soma and load data #####

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
        hemi = ['hemi-L','hemi-R']

        data = []
        for _,h in enumerate(hemi):
            gii_file = med_gii[0] if h == 'hemi-L' else  med_gii[1]
            print('using %s' %gii_file)
            data.append(np.array(surface.load_surf_data(gii_file)))

        data = np.vstack(data) # will be (vertex, TR)

        if idx == 0:
            median_sub = data[np.newaxis,:,:]
        else:
            median_sub = np.vstack((median_sub,data[np.newaxis,:,:]))


    print('computed median subject, from averaging %d subs'%median_sub.shape[0])
    median_data_all = np.median(median_sub,axis=0)

    return median_data_all

def make_median_soma_events(all_subs):
    # input sub list, file extension, out_dir, function to make median run
    
    # make function that makes median event data frame for x runs of sub or for median sub
    # now this will do
    onsets_allsubs = []
    durations_allsubs = []

    for idx,sub in enumerate(all_subs):
        # path to events
        eventdir = os.path.join(analysis_params['sourcedata_dir'],'sub-{sj}'.format(sj=str(sub).zfill(2)),'ses-01','func')
        print('event files from %s' % eventdir)

        # list of stimulus onsets
        events = [os.path.join(eventdir,run) for run in os.listdir(eventdir) if 'soma' in run and run.endswith('events.tsv')]
        events.sort()


        ##### median events df for each sub #####

        all_events = []
        for _,val in enumerate(events):

            events_pd = pd.read_csv(val,sep = '\t')

            new_events = []

            for ev in events_pd.iterrows():
                row = ev[1]   
                if row['trial_type'][0] == 'b': # if both hand/leg then add right and left events with same timings
                    new_events.append([row['onset'],row['duration'],'l'+row['trial_type'][1:]])
                    new_events.append([row['onset'],row['duration'],'r'+row['trial_type'][1:]])
                else:
                    new_events.append([row['onset'],row['duration'],row['trial_type']])

            df = pd.DataFrame(new_events, columns=['onset','duration','trial_type'])  #make sure only relevant columns present
            all_events.append(df)

        # make median event dataframe
        onsets = []
        durations = []
        for w in range(len(all_events)):
            onsets.append(all_events[w]['onset'])
            durations.append(all_events[w]['duration'])

        # append median event for sub, in all sub list
        onsets_allsubs.append(np.median(np.array(onsets),axis=0)) #append average onset of all runs
        durations_allsubs.append(np.median(np.array(durations),axis=0))

    # all subjects in one array, use this to compute contrasts
    events_avg = pd.DataFrame({'onset':np.median(np.array(onsets_allsubs),axis=0),'duration':np.median(np.array(durations_allsubs),axis=0),'trial_type':all_events[0]['trial_type']})
    print('computed median events, from averaging events of %d subs'%np.array(onsets_allsubs).shape[0])

    return events_avg    



def make_raw_vertex_image(data2plot,cmap,vmin,vmax,subject='fsaverage_gross'):
    ## function to fix web browser bug in pycortex ##
    # allows masking of data with nans #
    
    # INPUTS #
    # data2plot
    # cmap - string with colormap name
    # vmin
    # vmax
    # subjects
    
    # OUTPUT
    # vertex object to call in webgl 
    
    # Get curvature
    curv = cortex.db.get_surfinfo(subject)
    # Adjust curvature contrast / color. Alternately, you could work
    # with curv.data, maybe threshold it, and apply a color map. 
    curv.vmin = -1
    curv.vmax = 1
    curv.cmap = 'gray'
    
    # Create display data (Face, HANDS, legs)
    vx = cortex.Vertex(data2plot, subject, cmap=cmap, vmin=vmin, vmax=vmax)

    # Map to RGB
    vx_rgb = np.vstack([vx.raw.red.data, vx.raw.green.data, vx.raw.blue.data])
    curv_rgb = np.vstack([curv.raw.red.data, curv.raw.green.data, curv.raw.blue.data])

    # Pick an arbitrary region to mask out
    # (in your case you could use np.isnan on your data in similar fashion)
    alpha = ~np.isnan(data2plot) #(data < 0.2) | (data > 0.4)
    alpha = alpha.astype(np.float)

    # Alpha mask
    display_data = vx_rgb * alpha + curv_rgb * (1-alpha)

    # Create vertex RGB object out of R, G, B channels
    vx_fin = cortex.VertexRGB(*display_data, subject)

    return vx_fin
    

# function to align twin axis in same plot
def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)


# make function to retrieve estimates from list of subs and 
# return appended estimates
# to make proper ecc vs size plots for "subject median"


def append_pRFestimates(subdir,with_smooth=True,exclude_subs=['sub-07'],model='css',iterative=True):

####################
#    inputs
# subdir - absolute path to all subject dir (where fits are)
# with_smooth - boolean, use smooth data?
# exclude_subs=['subjects to be excluded from list']
#   outputs
# estimates - dictionary with appended estimated parameters

    allsubs = [folder for _,folder in enumerate(os.listdir(subdir)) if 'sub-' in folder]
    allsubs.sort()
    print('appending %d/%d subjects' %((len(allsubs)-len(exclude_subs)),len(allsubs)))
    print('using estimates from %s fit'%model)

    rsq = []
    xx = []
    yy = []
    size = []
    baseline = []
    beta = []
    ns = [] # for css
    new_subs = []

    exclude_counter = 0
    exclude_subs.sort() #then in order

    #load estimates and append
    for _,sub in enumerate(allsubs):

        if sub in exclude_subs[exclude_counter]:
            print('skipping %s, not included '%exclude_subs[exclude_counter])
            if len(exclude_subs)>(exclude_counter+1): exclude_counter += 1 # if more subs in list, increment counter
        else:

            # load prf estimates
            if with_smooth==True:    
                median_path = os.path.join(subdir,'{sj}'.format(sj=sub),'run-median','smooth%d'%analysis_params['smooth_fwhm'])
            else:
                median_path = os.path.join(subdir,'{sj}'.format(sj=sub),'run-median')

            if iterative==True: # if selecting params from iterative fit
                estimates_list = [x for x in os.listdir(median_path) if 'iterative' in x and x.endswith(model+'_estimates.npz')]
            else: # only look at grid fit
                estimates_list = [x for x in os.listdir(median_path) if 'iterative' not in x and x.endswith(model+'_estimates.npz')]
            estimates_list.sort() #sort to make sure pRFs not flipped

            estimates = []
            for _,val in enumerate(estimates_list) :
                print('appending %s'%val)
                estimates.append(np.load(os.path.join(median_path, val))) #save both hemisphere estimates in same array

            xx.append(np.concatenate((estimates[0]['x'],estimates[1]['x'])))
            yy.append(np.concatenate((estimates[0]['y'],estimates[1]['y'])))
               
            size.append(np.concatenate((estimates[0]['size'],estimates[1]['size'])))
            
            beta.append(np.concatenate((estimates[0]['betas'],estimates[1]['betas'])))
            baseline.append(np.concatenate((estimates[0]['baseline'],estimates[1]['baseline'])))
            
            if model=='css': ns.append(np.concatenate((estimates[0]['ns'],estimates[1]['ns']))) # exponent of css

            rsq.append(np.concatenate((estimates[0]['r2'],estimates[1]['r2'])))

            new_subs.append(sub) 
    
    estimates = {'subs':new_subs,'exclude_subs':exclude_subs,'r2':rsq,'x':xx,'y':yy,
                 'size':size,'baseline':baseline,'betas':beta,'ns':ns}

    return estimates


def make_raw_vertex2D_image(data2plot1,data2plot2,cmap,vmin,vmax,vmin2,vmax2,subject='fsaverage_gross'):
    ## function to fix web browser bug in pycortex ##
    # allows masking of data with nans #
    
    # INPUTS #
    # data2plot
    # cmap - string with colormap name
    # vmin
    # vmax
    # subjects
    
    # OUTPUT
    # vertex object to call in webgl 
    
    # Get curvature
    curv = cortex.db.get_surfinfo(subject)
    # Adjust curvature contrast / color. Alternately, you could work
    # with curv.data, maybe threshold it, and apply a color map. 
    curv.vmin = -1
    curv.vmax = 1
    curv.cmap = 'gray'
    
    # Create display data (Face, HANDS, legs)
    vx = cortex.Vertex2D(data2plot1,data2plot2, subject, 
                         cmap=cmap, vmin=vmin, vmax=vmax,
                        vmin2=vmin2, vmax2=vmax2)

    # Map to RGB
    vx_rgb = np.vstack([vx.raw.red.data, vx.raw.green.data, vx.raw.blue.data])
    curv_rgb = np.vstack([curv.raw.red.data, curv.raw.green.data, curv.raw.blue.data])

    # Pick an arbitrary region to mask out
    # (in your case you could use np.isnan on your data in similar fashion)
    #alpha = (data2plot1 == 0) | (data2plot2 == 0)
    #alpha = alpha.astype(np.float)

    # Alpha mask
    display_data = vx_rgb + curv_rgb #vx_rgb * alpha + curv_rgb * (1-alpha)

    # Create vertex RGB object out of R, G, B channels
    vx_fin = cortex.VertexRGB(*display_data, subject)

    return vx_fin
    

def make_2D_colormap(rgb_color='101',bins=50):
    # generate 2D basic colormap
    # and save to pycortex filestore
    
    ##generating grid of x bins
    x,y = np.meshgrid(
        np.linspace(0,1,bins),
        np.linspace(0,1,bins)) 
    
    # define color combination for plot
    if rgb_color=='101': #red blue
        col_grid = np.dstack((x,np.zeros_like(x), y))
        name='RB'
    elif rgb_color=='110': # red green
        col_grid = np.dstack((x, y,np.zeros_like(x)))
        name='RG'
    elif rgb_color=='011': # green blue
        col_grid = np.dstack((np.zeros_like(x),x, y))
        name='GB'
    
    fig = plt.figure(figsize=(1,1))
    ax = fig.add_axes([0,0,1,1])
    # plot 
    plt.imshow(col_grid,
    extent = (0,1,0,1),
    origin = 'lower')
    ax.axis('off')

    rgb_fn = os.path.join(os.path.split(cortex.database.default_filestore)[
                          0], 'colormaps', 'costum2D_'+name+'_bins_%d.png'%bins)

    plt.savefig(rgb_fn, dpi = 200)
       
    return rgb_fn  


##normalize data:
def normalize(M):
    return (M-np.nanmin(M))/(np.nanmax(M)-np.nanmin(M))


def equal_bin(arr,num_bins=4):
    # function to order array, divide in equally sized bins
    # and return array of bin labels (each vertex will be given a label value indicating the bin it belongs to)
    # essentially dividing in quantiles
    
    arr_sort = np.argsort(arr) # sort array and get indices
    bin_max_ind = (len(arr)/num_bins)*np.arange(1,num_bins+1) # max bin indices
    
    bin_counter = 0
    bin_labels = np.zeros(arr.shape)
    
    for k in range(len(arr)): # for all voxels

        bin_labels[arr_sort[k]]=bin_counter+0.5 # making my life easier in the colorbars (label will be mid position of bin range)

        if k==int(np.around(bin_max_ind[bin_counter]-1)): # if max index for bin reached, increment
            bin_counter +=1

    return bin_labels


# make function that takes array with shape (vertices,), with some soma estimate or relevant quantification4 plotting
# for whole brain and smooths it

def smooth_soma_array(header_dir,arr,out_dir,filestr,n_TR=141,file_extension='_sg_psc.func.gii'):
    # header_dir - dir to get an example gii file for header (need it to convert arr to gii before smoothing)
    # arr - array to smooth
    # out_dir - absolute path to save new gii array (non smoothed and smoothed)
    # filestr - string to add to name of new gii (identifier)
    
    # returns smoothed array
    
    smooth_filename = []
    
    for field in ['hemi-L', 'hemi-R']:
        
        num_vert_hemi = int(arr.shape[0]/2) #number of vertices in one hemisphere
        
        if field=='hemi-L':
            arr_4smoothing = arr[0:num_vert_hemi]
        else:
            arr_4smoothing = arr[num_vert_hemi::]
        
        # load run just to get header
        filename = [run for run in os.listdir(header_dir) if 'soma' in run and 'fsaverage' in run and field in run and run.endswith(file_extension)]
        filename.sort()
        
        img_load = nb.load(os.path.join(header_dir,filename[0]))
        
        # absolute path of new gii
        out_filename = os.path.join(out_dir,filestr+'_'+field+file_extension)
        
        print('saving %s'%out_filename)
        est_array_tiled = np.tile(arr_4smoothing[np.newaxis,...],(n_TR,1)) # NEED TO DO THIS 4 MGZ to actually be read (header is of func file)
        darrays = [nb.gifti.gifti.GiftiDataArray(d) for d in est_array_tiled]
        estimates_gii = nb.gifti.gifti.GiftiImage(header=img_load.header,
                                           extra=img_load.extra,
                                           darrays=darrays) # need to save as gii
        
        
        nb.save(estimates_gii,out_filename)

        _,smo_estimates_path = smooth_gii(out_filename,out_dir,fwhm=analysis_params['smooth_fwhm'])

        smooth_filename.append(smo_estimates_path)
        print('saving %s'%smo_estimates_path)
        
    # load both hemis and combine in one array
    smooth_arr = []
    for _,name in enumerate(smooth_filename): # not elegant but works
        img_load = nb.load(name)
        smooth_arr.append(np.array([img_load.darrays[i].data for i in range(len(img_load.darrays))]))
    out_array = np.concatenate((smooth_arr[0][0],smooth_arr[1][0]))  
        
    return out_array