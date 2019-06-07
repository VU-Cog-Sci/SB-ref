#useful functions to use in other scripts

def median_gii(files,outdir):
    
    ##################################################
    #    inputs:
    #        files - list of absolute filenames to do median over
    #        outdir - path to save new files
    #    outputs:
    #        median_file - absolute output filename
    ##################################################
    
    import re
    import nibabel as nb
    import numpy as np
    import os
    
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
    
    import imageio
    from skimage import color
    import numpy as np
    import cv2
    from skimage.transform import rescale
    from skimage.filters import threshold_triangle
    import os
    
    im_gr_resc = np.zeros((len(filenames),int(screen[1]*scale),int(screen[0]*scale)))
    
    for i, png in enumerate(filenames): #rescaled and grayscaled images
        im_gr_resc[i,:,:] = rescale(color.rgb2gray(imageio.imread(png)), scale)
    
    img_bin = np.zeros(im_gr_resc.shape) #binary image, according to triangle threshold
    for i in range(len(im_gr_resc)):
        img_bin[i,:,:] = cv2.threshold(im_gr_resc[i,:,:],threshold_triangle(im_gr_resc[i,:,:]),255,cv2.THRESH_BINARY_INV)[1]
    
    # save as numpy array
    np.save(outfile, img_bin.astype(np.uint8))


def highpass_gii(filenames,polyorder,deriv,window,outpth):
    
    import os
    from nilearn import surface
    from scipy.signal import savgol_filter
    
    filenames_sg = []
    filenames.sort() #to make sure their in right order
    
    for run in range(len(filenames)//2): # for every run (2 hemi per run)

        run_files = [x for _,x in enumerate(filename) if 'run-0'+str(run+1) in os.path.split(x)[-1]]

        data_both = []
        for _,hemi in enumerate(run_files):
            data_both.append(surface.load_surf_data(hemi).T) #load surface data

        data_both = np.hstack(data_both) #stack then filter
        print('filtering run %s' %run_files)
        data_both -= savgol_filter(data_both, window, polyorder, axis=1,deriv=deriv)
        
        filenames_sg.append(data_both)
    
    return np.array(filenames_sg)

def highpass_confounds(confounds,nuisances,polyorder,deriv,window,tr,outpth):
    
    import os
    import pandas as pd
    from spynoza.filtering.nodes import savgol_filter_confounds
    from sklearn.decomposition import PCA

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

    
