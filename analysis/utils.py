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
    
    for i, png in enumerate(png_filename): #rescaled and grayscaled images
        im_gr_resc[i,:,:] = rescale(color.rgb2gray(imageio.imread(png)), scale)
    
    img_bin = np.zeros(im_gr_resc.shape) #binary image, according to triangle threshold
    for i in range(len(im_gr_resc)):
        img_bin[i,:,:] = cv2.threshold(im_gr_resc[i,:,:],threshold_triangle(im_gr_resc[i,:,:]),255,cv2.THRESH_BINARY_INV)[1]
    
    # save as numpy array
    np.save(outfile, img_bin.astype(np.uint8))
