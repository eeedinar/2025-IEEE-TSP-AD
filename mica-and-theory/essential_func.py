#!/usr/bin/env python3
"""
Author: Abdullah Al Bashit
Ph.D. Student in Electrical Engineering
Northeastern University, Boston, MA
Date: 01/01/2021
"""
########################### ---------- Essential functions ---------- ###########################

## import packages
import os, shutil, h5py, time, json, cv2, copy, glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf                        # pdf generation package
import ipywidgets 
from PyPDF2 import PdfFileMerger
import scipy.stats as stats
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable       # for show_colorbar function
import matplotlib.image as mpimg
from IPython.display import display, clear_output
from scipy.ndimage import label

# Lin Yang's BNL packages for 1-d averaging
from py4xs.hdf import h5xs,h5exp,lsh5
from py4xs.data2d import Data2d    

## create 1d data from Lin Yang's code (py4xs packages are used here)
def azimuthal_averaging(file, qgrid, n_proc=8, exp_folder = ""):
    """
        azimuthal_averaging(masked_file, qgrid, n_proc=8)
    """
    de = h5exp(exp_folder+"exp.h5")

    dt  = h5xs(file, [de.detectors, qgrid])
    tic = time.perf_counter()
    print(f'Circular averaging starts now ... ')
    dt.load_data(N=n_proc)
    print(f'{file} total 1-d averaging time {time.perf_counter() - tic} seconds')


### dropdown name to directory
def dropdown_to_abs_dir(dropdown_name, csv_file_location, samples_csv):
    ## 'Mar-2023-Sample#1971' --> '/Volumes/HDD/BNL-Data/Mar-2023/1971/'
    df = pd.read_csv(os.path.join(csv_file_location,samples_csv))
    idx = df[df["dropdown-name"]==dropdown_name].index
    return df['bnl-scan-sample-dir'][idx].values[0]


### read csv file to change current python directory
def change_python_path(dropdown_name, csv_file_location, samples_csv):
    

    df = pd.read_csv(os.path.join(csv_file_location,samples_csv))
    idx = df[df["dropdown-name"]==dropdown_name].index

    print(f'Python directory is set to load samples from : {df["dropdown-name"][idx].values[0]}')
    dropdown_name_list = df["dropdown-name"].values
    os.chdir(df['bnl-scan-sample-dir'][idx].values[0])
    exp_folder = df['corresponding-exp-dir'][idx].values[0]

    return dropdown_name, dropdown_name_list, os.getcwd(), exp_folder

## np.arange(start, stop, step) funcation but stop value is inclusive
def drange(start, stop, step):
    """
        np.fromiter(drange(0, 10, 1), float)
    """
    i = start
    while i<=stop:
        yield i
        i +=step

def h5File_h5Dir_csv_loc_by_h5file(file, BNL_dir, sub_dir):
    """ 
        h5File_h5Dir_csv_loc_by_h5file( file = "1948_HIPPO-roi1_0_0_masked_intp.h5",  
                                    BNL_dir     = '/Volumes/HDD/BNL-Data/Mar-2023' ,
                                    sub_dir     = "CSV_Conv-8-point")     
    """
    for file_search in glob.iglob(f'{BNL_dir}/**/*', recursive=True):
        if file_search.find(file) > -1 and file_search.endswith(".h5"):
            directory = os.path.dirname(file_search)
        if file_search.find(file) > -1 and f'/{sub_dir}/' in file_search:
            csv_file = file_search
            break
    return file, directory, csv_file


### pixalated sum function
def pixalated_sum_waxs(file, save_as_file = False, save_as_file_only = False):
    '''
        does not create file if it already exists, otherwise create directory (if not exist) and file
        dset_waxs_sum = pixalated_sum_waxs(file, save_as_file=True)   --> saves file as npz
    '''
    
    folder = 'pixalated_sum_waxs'
    new_file = f'{folder}/{file.strip(".h5")}-pixalated_sum_waxs.npz'

    ### if file already exists just return function
    if save_as_file_only and os.path.isfile(new_file):
        return


    with h5py.File(file,'r') as hdf:
        dset_waxs = np.array(hdf.get(f'{h5_top_group(file)}/primary/data/pilW2_image'))         # (4941, 1043, 981)
        dset_waxs_sum = np.sum(dset_waxs,axis=0)                                # (1043, 981) summing over each pixel of all frames

    dset_waxs_sum_stat = 'Pixalated Sum WAXS shape = ', dset_waxs_sum.shape , \
              'Min. = {:02f}'.format(dset_waxs_sum.min()), \
              'Mean = {:02f}'.format(dset_waxs_sum.mean()) , \
              'Median = {:02f}'.format(np.median(dset_waxs_sum)) , \
              'Max. = {:02f}'.format(dset_waxs_sum.max())
    print(dset_waxs_sum_stat)
    dset_waxs_sum_df = pd.DataFrame({'percentile' : np.arange(70,100) , 'value' : [np.percentile(dset_waxs_sum, i) for i in range(70,100)] })
    dset_waxs_sum_df['value'] = dset_waxs_sum_df['value'].map("{:,.2f}".format)
    print(dset_waxs_sum_df)


    if save_as_file:
                
        if os.path.isfile(new_file):
            print(f'DID not createe file {new_file}, already exists')

        elif not os.path.exists(folder):
            os.makedirs(folder)
        
        if not os.path.isfile(new_file):
            np.savez(new_file, waxs_sum = dset_waxs_sum, waxs_sum_stat = dset_waxs_sum_stat, waxs_sum_percentile = dset_waxs_sum_df, allow_pickle=True)

    return dset_waxs_sum


# github repo to create discrete cmap https://gist.github.com/jakevdp/91077b0cae40f8f8244a
def discrete_cmap(N, base_cmap=None):
    """ 
    Use of this function :
        f = plot_heat_map_from_file(file, qgrid2, scatterings = scatterings, cmap = discrete_cmap(N=5, base_cmap = 'cubehelix'))
    """
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


### given qgrid value find index of an array
def qgrid_to_indices(qgrid, qvalue):
    """
        idx = qgrid_to_indices(qgrid, qvalue)    # idx = 310 for qvalue=1.3
    """
    return np.argmin(qgrid < qvalue)

### valid indices search function
def h5_top_group(file):
    """ 
        input argument string of a h5py file name
        returns seperator
        seperator = h5_top_group(file)
    """
    return file.split('_masked')[0] if file.find('_masked') >= 0 else file.split('.')[0]

def valid_idx_search(qgrid, Iq, show_q = False):
    """
        Iq shape (#frames, #1-d avg. values) ex. (3721,690) shape must be min. (1,690)

        function call: 
        idx_l, idx_u, valid_diff_values = valid_idx_search(qgrid2, Iq)

        retrun:
            idx_l = first valid   indices    SAXS = 2,   WAXS = 109 , merged = 0
            idx_u = first invalid indices    SAXS = 125, WAXS = 579 , merged = 690
            valid_diff_values = (#frames, only valid values)  ex. 
            '_SAXS'  = (4941, 123) --> 125 - 2 = 123
            '_WAXS2' = (4941, 470) --> 579-109 = 470
    """
    list_ = ~np.isnan(Iq[0,:].flatten())                      # list_ = [False, False, True, True, ...] boolean list in the first frame 
    idx_l = np.argmax(list_==True)                            # find first occurance of a boolean True ; SAXS = 2,   WAXS = 109 , merged = 0
    idx_u  = len(list_) - np.argmax(list_[::-1]==True)        # find last occurance of a invalid number;   SAXS = 125 , WAXS = 579 , merged = 690 valid on 124, 578 , 689 indices
    if show_q:
        print(f'low valid idx = {idx_l}, low valid Q = {qgrid[idx_l]:0.3f}, high valid idx = {idx_u-1} , high valid Q = {qgrid[idx_u-1]:0.3f}')
    
    valid_diff_values = Iq[:, idx_l:idx_u]                 # SAXS-Iq --> (1,2:125,:3721)
    
    return idx_l, idx_u, valid_diff_values

### find representative value for each frame
def find_rep_value(qgrid, Iq, args=None, method = 'polyfit'):
    """
    Iq (3721,690) may contain valid or invalid values
    for polynomial averaging:
        method = 'polynomial' 
    for just simply averaging:
        method = 'circ'
    diff_patterns = find_rep_value(qgrid2, Iq, args, method = 'polyfit')
    diff_patterns = find_rep_value(qgrid2, Iq, method = 'circ')
    diff_patterns = find_rep_value(qgrid2, Iq , args=1.34, method = 'point')
    return diff_patterns = (3721)
    
    """
    n_patterns = len(Iq)

    if method == 'polyfit':
        poly_ord, idx_l, idx_u, limit_l, limit_u = args
        diff_patterns = [];
        for frame in range(n_patterns):                      # range(100) --> look for first 100 frames

            X = qgrid[idx_l:idx_u]
            y = Iq[frame][idx_l:idx_u]

            # Find coefficients
            coefs = np.polyfit(X,y,poly_ord)
            #print(coefs)

            X_test = X[limit_l:limit_u]
            y_test = y[limit_l:limit_u]
            diff_patterns.append(np.max( y_test - np.polyval(coefs, X_test) ));        # (0,2:125,0)  reshape diff_patterns shape (0, 2:125, 3721)
        diff_patterns = np.array(diff_patterns)      
    
    elif method == 'circ':
        _ , _ , valid_diff_values = valid_idx_search(qgrid, Iq, show_q = False)
        diff_patterns = np.zeros(len(valid_diff_values));
        
        for frame in range(n_patterns):                      # range(100) --> look for first 100 frames
            diff_patterns[frame] = np.nanmean(valid_diff_values[frame])  # calculate the mean of array ignoring the NaN value

    elif method == 'point':
        qvalue = args
        idx = qgrid_to_indices(qgrid, qvalue=qvalue)
        diff_patterns = np.zeros(len(Iq));

        for frame in range(n_patterns):                      
            diff_patterns[frame] = Iq[frame, idx]  # calculate the mean of array ignoring the NaN value
        print(f'qvalue is : {qvalue:0.4f}')
    print(f'Minimum, Maximum Iq : {np.min(diff_patterns):0.4f}, {np.max(diff_patterns):0.4f}')
    return diff_patterns

### snaking function
def snaking(Width, Height, X=None,):
    """
        img_orig = snaking(Width, Height, diff_patterns)
    """
    X = copy.deepcopy(X)    # deep copy
    if X is None:
        X_coor = np.arange(Height*Width).reshape(Height,Width)
    else:
        X_coor = X.reshape(Height,Width)            # input id numpy reshaped as height * width

    img_orig = np.flipud(X_coor)                                     # flipud matrix
    idx_even = np.arange(Height-2,-1,-2)                             # Get odd indices to leave to as it is
    img_orig[idx_even,:] = np.fliplr(img_orig[idx_even])             # flipping left to right 
    return img_orig

### snaking width_height extraction
## attrs in h5py is recorded in json format

def width_height(file, directory=None):
    """
        Width, Height = width_height(file)
    """
    directory = os.getcwd() if directory == None else directory
    with h5py.File(os.path.join(directory,file),'r') as hdf:
        dset = hdf.get(h5_top_group(file))
        header = json.loads(dset.attrs['start'])
        Height, Width = header['shape']
    return Width, Height

### read Iq data from hdf file
def read_Iq(file, scattering, frame = None, directory = None):
    """
        read Iq data
        Iq = read_Iq(file, scattering)
        Iq = read_Iq(file, scattering, frame)
    """
    directory = os.getcwd() if directory == None else directory
    with h5py.File(os.path.join(directory, file),'r') as hdf:
        Iq = hdf.get(f'{h5_top_group(file)}/processed')              # Iq = hdf.get('2048_B16/processed')
        Iq = np.array(Iq.get(scattering))                            # Iq = np.array(2048_B16/processed/merged')
        Iq = Iq[:,0,:] if frame ==None else Iq[frame,0,:]            # Iq shape (3721, 690) but if frame is given Iq shape (690,) 

    return Iq


### data binning/ discritizing data and return heatmap matrix
def discritize_scattering(file, qgrid, scattering, heatmap_rep_value = 'circ', args = None,  data_binning=False, bins=None, directory=None):
    """
        img_orig =  discritize_scattering(file, qgrid, scattering, data_binning=False, bins=np.fromiter(drange(0, saxs_max, saxs_inc), float) )
    """        

    directory = os.getcwd() if directory == None else directory
    Width, Height = width_height(file, directory)

    Iq = read_Iq(file, scattering, directory=directory)      # Iq shape (3721, 690)

    if heatmap_rep_value == 'circ' or heatmap_rep_value == 'polyfit':
        diff_patterns = find_rep_value(qgrid, Iq, method = heatmap_rep_value)  
    else :
        diff_patterns = find_rep_value(qgrid, Iq, args=args, method = heatmap_rep_value)
    
    # check for data binning/bucketing (returns inds on the right side of the interval it lies) - e.g. x = np.array([[5,6], [-1,0]]) ; bins = np.arange(0,5,1) ; inds = np.digitize(x, bins, right=False); print(bins,'\n' ,inds)
    if data_binning: 
        inds = np.digitize(diff_patterns, bins, right=False)    # right=False, right side of bin edge will be excluded        
        # x    = np.array([1.2, 10.0, 12.4, 15.5, 20., -1, 5, 30]); bins = np.array([0, 5, 10, 15, 20]); inds = np.digitize(x,bins,right=False) ; inds[inds == len(bins)] = len(bins) - 1; print(bins[inds])
        inds[inds == len(bins)] = len(bins) - 1   # inds for bins that are out of bound is being restricted to the maximum bin index (len(bins) - 1)
        diff_patterns = bins[inds]   

    else:
        pass

    img_orig = snaking(Width, Height, diff_patterns)

    return img_orig

### generate heatmap for differnet scatterings
def plot_heat_map_from_data(img_orig, Width, Height, args, title= None, cmap="viridis", norm=None, ticks=None, alpha=None):
    """
        plot_heat_map_from_data(img_orig, Width, Height, args = None, title= None, cmap="viridis")
    """
    ########## --------- matplotlib mouse hovering function for snaking --------- ##########
    frame_cor = snaking(Width, Height)                  # snaking indices for heat map, 
    numrows, numcols = Height, Width                    # format_coord function requires this global variables
    def format_coord(x, y):
        col = int(x)                                  # truncate x values
        row = int(y)                                  # truncate y values
        if 0 <= col < numcols and 0 <= row < numrows:
            z = np.flipud(frame_cor)[row, col]        # flipping to get correct value of z
            return 'x=%1.2f, y=%1.2f, FRAME=%d' % (x, y, z)
        else:
            return 'x=%1.2f, y=%1.2f' % (x, y)        # outside the plotting range, no need

    # def mouse_event(event):
    #     x, y = event.xdata, event.ydata
    #     col = int(x)                                  # truncate x values
    #     row = int(y)                                  # truncate y values
    #     if 0 <= col < numcols and 0 <= row < numrows:
    #         z = np.flipud(frame_cor)[row, col]        # flipping to get correct value of z     
    #         print('FRAME:', z)
    #     else:
    #         print('NO FRAME')        # outside the plotting range, no need
    
    # ### plotting
    f, ax = args
    # cid   = f.canvas.mpl_connect('button_press_event', mouse_event)
    ax.clear()
    ax.autoscale(True)
    im    = ax.imshow(img_orig, cmap = cmap, interpolation = 'none', origin='upper', extent=[0,Width,0,Height], aspect='equal', norm=None, alpha=alpha)
    show_colorbar(im,f,ax, ticks=ticks)
    ax.format_coord = format_coord
    ax.set(title = title, xticks = (np.arange(0,Width,5)), yticks = (np.arange(0,Height,5))) #


### plotting heatmap from file
def plot_heat_map_from_file(file, qgrid, scatterings = None, heatmap_rep_value = 'circ', arg_qvalue = None, cmap="viridis", args = None, data_binning=False, bins = None, alpha=None):
    """
        Input args:
            must be tupple scattering = ('_SAXS',)
            args = (f, axs)
        f = plot_heat_map_from_file(file, qgrid, scatterings = scattering, cmap="viridis")
        f = plot_heat_map_from_file(file, qgrid, scatterings = None, cmap="viridis")
        f = plot_heat_map_from_file(file, qgrid2, scatterings = scatterings, cmap = 'viridis', args = (f,ax), data_binning=True, bins = np.array([0,10,20, 30]))
    """
    ## No given scattering argument plot all SAXS, WAXS, Merged
    if scatterings == None:
        scatterings = ('_SAXS',          '_WAXS2',        'merged'    )
    
    f, axs = plt.subplots(1, len(scatterings), num=f'{file} Heat maps', figsize=(10,5)) if args == None else args

    if len(scatterings) == 1:   
        axs =  [axs]       # for one scatterings input making axs a list to use in the loop as axs[i]
        bins = [bins]
    ### mouse hovering function call
    Width, Height = width_height(file)
    
    img_orig = np.zeros((len(scatterings), Height, Width))
    for i, scattering in enumerate(scatterings):

        img_orig[i] = discritize_scattering(file, qgrid, scattering, heatmap_rep_value, arg_qvalue, data_binning, bins[i]) if heatmap_rep_value == 'point' else discritize_scattering(file, qgrid, scattering, heatmap_rep_value = 'circ', args = None, data_binning=data_binning, bins=bins[i])
        
        plot_heat_map_from_data(img_orig[i], Width, Height, args = (f, axs[i]), title= f'{scattering} {file}', cmap=cmap, alpha=alpha)
    #plt.tight_layout()
    
    # plt.show()

    ### to avoild multiple plots in jupyter-lab
    # display(f)
    # clear_output(wait=True)

    return f, img_orig[0] if len(scatterings)==1 else img_orig

def cwd_files_search_with(seek_str, search_where = 'end', directory = None):
    """
        files_sorted = cwd_files_search_with('.h5')
    """
    directory = os.getcwd() if directory == None else directory
    if os.path.isdir(directory):
        files = []
        if search_where == 'end':
            for file in [each for each in os.listdir(directory) if each.endswith(seek_str)]:
                files.append(file)
        
        elif search_where == 'start':
            for file in [each for each in os.listdir(directory) if each.startswith(seek_str)]:
                files.append(file)

        files_sorted = sorted(files)
        return files_sorted
    else:
        return []

def plot_all_heat_maps_cwd(file, qgrid, scatterings, seek_str):
    """
        scatterings = ('_SAXS', '_WAXS2')
        
        Function call:
            plot_all_heat_maps_cwd('output.pdf', qgrid2, scatterings, seek_str='_masked.h5')
    """
    pdf = matplotlib.backends.backend_pdf.PdfPages('output.pdf')
    files_sorted = cwd_files_search_with(seek_str)
    for file in files_sorted:
        print(f'Loading file {file}')
        
        try:       
            f = plot_heat_map_from_file(file, qgrid, scatterings = scatterings ) # figure object
            print(f'{file} Task Finished. Figure Number = ' , f.number)
        except:    
            continue
          
        pdf.savefig(f);   #f.savefig("foo.pdf", ) 

    pdf.close();
    print('PDF creation Finished')

### load config.json file in python
def get_json_str_data(file):
    """
        data = get_json_str_data("config.json")
    """
    with open(file, 'r') as f:
        data = json.load(f)         # read data from file to python object
        data = json.dumps(data)     # convert python obj to json format data
        data = json.loads(data)     # convert json formatted string into python dictionary
    return data

### set patch attributes
def set_patch_attributes(file, args, method = 'thresholding'):
    """
        set_patch_attributes(file, patches)
        set_patch_attributes(file, patches, method = 'thresholding')
    """
    with h5py.File(file,'r+') as hdf:
        
        dset = hdf.get(f'{h5_top_group(file)}/processed')              # Iq = hdf.get('2048_B16/processed')
        if method == 'rec_circ_patch':
            dset.attrs['patches'] = json.dumps(args)
        elif method == 'thresholding':
            dset.attrs['threshold'] = json.dumps(args)    # here args = [amin, amax, threshold]
        elif method == 'thr_rec_circ_patch':

            thr_args, patches_arg = args
            thr_args = thr_args[1:]                  # dset_waxs_sum omitted, thr_args = [amin, amax, threshold]
            dset.attrs['threshold'] = json.dumps(thr_args)    # here args = [amin, amax, threshold]
            dset.attrs['patches'] = json.dumps(patches_arg)


    return "patching information written on the h5 file processed directory"

### get patch attributes
def get_patch_attributes(file, method = 'thresholding'):
    """
        patches = get_patch_attributes(file)
        patches = get_patch_attributes(file, method = 'thresholding')
    """
    with h5py.File(file,'r') as hdf:
        
        dset = hdf.get(f'{h5_top_group(file)}/processed')              # Iq = hdf.get('2048_B16/processed')

        if method == 'rec_circ_patch':
            patches = json.loads(dset.attrs['patches'])    

        elif method == 'thresholding':
            patches = json.loads(dset.attrs['threshold'])  
        
        elif method == 'thr_rec_circ_patch':
            patches = json.loads(dset.attrs['threshold']) , json.loads(dset.attrs['patches'])
    
    return patches


def file_polyfit_heatmap_plot(file, indices, qgrid2):
    """
        file = '2048_B8.h5'
        ### q_low, q_high, q_low - idx[2] and q_high + idx[2] for poly fit, number of points
        indices = ((0.115, 0.16, 13,   48, '_SAXS' , 'bumpy'),      # 72, 81
                   (1.48,  1.57, 10, 4.15, '_WAXS2', 'paraffin'),   # 345, 355
                   (1.30,  1.38, 10,  4.7, '_WAXS2', 'amyloid'),    # 309, 323
                   (1.89,  1.93,  7,  3.2, '_WAXS2', 'mica')        # 427, 433
                  )

        file_polyfit_heatmap_plot(file, indices, qgrid2)
    """

    # polyfit and q value function
    f, axs = plt.subplots(2,len(indices), figsize = (16,9), num = f'{file}')

    def polyfit_heatmap_plot(frame_polyfit, poly_ord = 1):

        for idx_indices,index in enumerate(indices):

            ### extracting values from the patches
            idx_l_t = np.max(np.where(qgrid2 <= index[0]))  # 73
            idx_u_t = np.max(np.where(qgrid2 <= index[1]))  # 81
            idx_l =  idx_l_t - index[2]                     # 73 - 13 = 60
            idx_u =  idx_u_t + index[2]                     # 81 + 13 = 90
            limit_l = index[2]                              # 13
            limit_u = index[2] + idx_u_t - idx_l_t          # 13 + 81 -73 = 21
            scattering = index[4]
            comment = index[5]
            #print(idx_l, idx_u, limit_l, limit_u)

            ### read Iq data
            Iq = read_Iq(file, scattering)    # read Iq data

            X = qgrid2                        # X is qgrid value
            y = Iq[frame_polyfit]             # One frame Iq values

            #print(idx_l, idx_u, X[idx_l], y[idx_l], X[idx_u-1] , y[idx_u-1])
            X = qgrid2[idx_l:idx_u]           # get valid Iq values q rannge
            y = y[idx_l:idx_u]                # get valid Iq rannge

            coefs = np.polyfit(X,y,poly_ord)  # fit polynomial

            X_test = X[limit_l:limit_u]       # extract the narrow region for X
            y_test = y[limit_l:limit_u]       # extract the narrow region for y
            
            max_ind = np.argmax( y_test - np.polyval(coefs, X_test))
            print(f'{comment} max-point to fitting-point difference:  {np.max( y_test - np.polyval(coefs, X_test))}') 
            
            ## plot polynomial and heat map
            axs[0, idx_indices].clear()
            axs[0, idx_indices].scatter(X,y, color='red' , label = 'exp data') 
            axs[0, idx_indices].scatter(X_test, y_test, color='purple' , label = 'fit data')
            axs[0, idx_indices].scatter(X, np.polyval(coefs, X), color = 'black' , label = 'polyfit')
            axs[0, idx_indices].scatter(X_test[max_ind], y_test[max_ind], color = 'blue', marker='^', label = 'max point')
            axs[0, idx_indices].scatter(X_test[max_ind], np.polyval(coefs, X_test[max_ind]), color = 'orange' , marker=r'$\clubsuit$' ,label = 'ref point')
            axs[0, idx_indices].annotate("", (X_test[max_ind], y_test[max_ind]) , (X_test[max_ind], np.polyval(coefs, X_test[max_ind])), arrowprops={'arrowstyle':'<-'})
            axs[0, idx_indices].set(title = f'Poly_ord {poly_ord} {qgrid2[idx_l_t+max_ind]:.2f} - {frame_polyfit} {comment}' ,
                        xlabel = 'X' , ylabel = 'y', xscale='linear', yscale = 'linear')
            axs[0, idx_indices].legend()

            ## gather heat map data
            args = poly_ord, idx_l, idx_u, limit_l, limit_u
            diff_patterns = find_rep_value(qgrid2, Iq, args, method = 'polyfit')

            # snaked diff. patterns and plot heat map
            img_orig = snaking(Width, Height, diff_patterns)   
            plot_heat_map_from_data(img_orig, Width, Height, args = (f,axs[1, idx_indices]), title = (f'Q~{index[0], index[1]} {scattering} {index[3]}$\AA$'), )

    #        f.colorbar(pos, ax= axs[1, idx_indices])

        plt.suptitle(f'{h5_top_group(file)} Polyfit and Heat maps')
        plt.tight_layout()
        plt.show()
    
    ### frame no. and polyfit Int slider
    Width, Height = width_height(file)
    n_patterns = Width*Height

    frame = ipywidgets.IntSlider(min=0, max=n_patterns-1, value=0, description="Frame", continuous_update=False)
    poly_ord = ipywidgets.IntSlider(min=1, max=10, value=1, description='Poly fit order', continuous_update=False)
    ipywidgets.interact(polyfit_heatmap_plot, frame_polyfit=frame, poly_ord = poly_ord)

    ### save pdf widgets
    ipywidgets.interact_manual.opts['manual_name'] = 'Save PDF'
    @ipywidgets.interact_manual()
    def foo():
        f.savefig(f"{file}.pdf")

def pdfs_merging(directory = '', output = 'result.pdf'):
    """
        merge all pdfs in the current directory :
            directory = '', output = 'result.pdf')
        merge all pdfs in the /PDF directory : 
            directory = '/PDF', output = 'result.pdf')
    """
    root_dir = os.getcwd()
    print(f'Current location {root_dir}')
    try:
        print(f'Changing to {root_dir}/{directory} directory ')
        os.chdir(f'{root_dir}/{directory}')
        pdfs = cwd_files_search_with('.pdf')
        merger = PdfFileMerger()

        for pdf in pdfs:
            merger.append(pdf)
        
        merger.write(output)
        merger.close()
        print(f'{output} file written successfully')
        os.chdir(root_dir)
    except:
        print('Either wrong directory or merging operation failed')
        os.chdir(root_dir)

    print('Back to root directory ', os.getcwd())

def show_colorbar(im,f,ax, position="right", ticks=None):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(position, size="2%", pad=0.05)
    cax.clear()
    f.colorbar(im, cax=cax, orientation='vertical', ticks=ticks)

def global_thresholding(input_array, thr, binary_inv = False):
    """
        gray_img, thr_cond = global_thresholding(dset_waxs_thr, thr, binary_inv = True)
    """
    if binary_inv:
        #### binary inversion
        gray_img = np.zeros_like(input_array);      # initialize a zero matrix like dset_waxs_thr - (gray matrix)
        thr_cond = input_array>thr;                 # do thresholding (above thr it goes to zero)
        gray_img[thr_cond] = 0;                       # make zero/black to each elements where condition is true
        gray_img[np.invert(thr_cond)] = 255;          # make 255/white to each elements where condition is inverted/false
        gray_img = gray_img==255                      # thresholded image is now binary image 

    else:
        #### binary only
        gray_img = np.zeros_like(input_array);      # initialize a zero matrix like dset_waxs_thr - (gray matrix)
        thr_cond = input_array<thr;                 # do thresholding (above thr it goes to zero)
        gray_img[thr_cond] = 0;                       # make zero/black to each elements where condition is true
        gray_img[np.invert(thr_cond)] = 255;          # make 255/white to each elements where condition is inverted/false
        gray_img = gray_img==255                      # thresholded image is now binary image 

    return gray_img, thr_cond



def loading_dset_waxs_sum(file, load_from = 'npz', show_stat=False):
    """
        dset_waxs_sum = loading_dset_waxs_sum(file, load_from = 'npz')
        load_from = 'npz' # 'h5'
    """
    ### semi spec
    folder = 'pixalated_sum_waxs'
    if load_from == 'h5':
        ##### Load file before WAXS thresholding (Requires to run the folliwing cell only) - a 300x300 roi should not take more than 3 min to load for M1 macbook
        dset_waxs_sum = pixalated_sum_waxs(file, save_as_file=False)

    elif load_from == 'npz':
        ### load pixalated sum file (2048_B8.h5-pixalated_sum_waxs.npz) from the saved folder (pixalated_sum_waxs)
        try:
            print(file)
            arr = np.load(f'{folder}/{file.strip(".h5")}-pixalated_sum_waxs.npz', allow_pickle=True)
            dset_waxs_sum      = arr['waxs_sum']
            print('dset_waxs_sum was successful')
            dset_waxs_sum_stat = arr['waxs_sum_stat']
            dset_waxs_sum_df   = arr['waxs_sum_percentile']
            if show_stat:
                print(dset_waxs_sum_stat)
                print(dset_waxs_sum_df)
        except:
            print(f'{folder}/{file.strip(".h5")}-pixalated_sum_waxs.npz file not found')
    else:
        print('Something Went Wrong')
    return dset_waxs_sum


def threshold_patch_one_frame(dset_waxs, args):
    """
        args = dset_waxs_sum, a_min, a_max, thr
        dset_waxs_thr, gray_img, thr_fr_img = threshold_patch_one_frame(dset_waxs, args)
    """
    dset_waxs_sum, a_min, a_max, thr = args
    #### thresholded pixalated sum
    dset_waxs_thr = np.clip(dset_waxs_sum,a_min,a_max)            # image thresholding    

    #### global thresholding 
    gray_img, thr_cond = global_thresholding(dset_waxs_thr, thr, binary_inv=True)

    #### Plot image after thresholding  
    thr_fr_img = dset_waxs*gray_img               # thresholded image by the binary mask
    thr_fr_img[thr_cond] = -1;                    # thresholded position replaced by -1
    
    return dset_waxs_thr, gray_img, thr_fr_img

def rec_circ_patch_one_frame(img, patches):
    """
        dset_waxs[0] = rec_circ_patch_one_frame(dset_waxs[0], patches)
    """
    for args in patches:   
        if type(args[1]) == int: 
            orig, radius = args
            ### Syntax: cv2.circle(image, center_coordinates, radius, color, thickness/(-1 = fill by the color black=0)
            img = cv2.circle(img, tuple(orig), radius, -1, -1)
        elif type(args[1]) == list:    
            starting_point, ending_point = args
            ### Syntax: cv2.rectangle(image, starting_point, ending_point, color, thickness)
            img = cv2.rectangle(img, tuple(starting_point), tuple(ending_point), -1, -1)
    return img

def threshold_rec_circ_patch_one_frame(dset_waxs, args):
    """
        args = dset_waxs_sum, a_min, a_max, thr
        dset_waxs_thr, gray_img, thr_fr_img = threshold_patch_one_frame(dset_waxs, args)
    """
    threshold_args, rec_circ_patch_args   = args
    dset_waxs_thr, gray_img, thr_fr_img = threshold_patch_one_frame(dset_waxs, threshold_args)

    _, _, thr_fr_img = threshold_patch_one_frame(dset_waxs, threshold_args)
    img = rec_circ_patch_one_frame(thr_fr_img, rec_circ_patch_args)

    return img

def patching(file, frame, qgrid, args, axes=None, method = 'rec_circ_patch', exp_folder = os.getcwd()):
    """
    method = 'rec_circ_patch' 
             'thresholding'
             'thr_rec_circ_patch'

    thresholding with axes: 
        patching(file, frame, qgrid2, args=args, axes = (f,[axs[1,0], axs[1,1]]), method = 'thresholding',)
    Individual frame thresholding:
        patching(file, frame.value, qgrid2, args=args, method = 'thresholding')
    Rectangual Circular patching:
        patching(file, frame, qgrid2, args=patches, method = 'rec_circ_patch')

    return:
        Iq_M = returns Iq values of Masked frame
    """
    ## semi-specs
    valid_range_min, valid_range_max = (-1,10)      # just for WAXS Display purpose
    scattering = '_WAXS2'
    masked_file = f'{h5_top_group(file)}_masked_{frame}.h5'

    ## computation
    with h5py.File(file,'r') as hdf:
        dset = hdf.get(f'{h5_top_group(masked_file)}/primary/data')
        dset_saxs_no_mask = np.expand_dims(dset['pil1M_image'][frame], axis=0)
        dset_waxs = np.expand_dims(dset['pilW2_image'][frame], axis=0)
        print(f'frame information extraction completes with _SAXS shape {dset_saxs_no_mask.shape} _WAXS shape {dset_waxs.shape}...')

    # create temporary file
    if os.path.isfile(masked_file):
        os.remove(masked_file)

    # overwriting patching information
    with h5py.File(masked_file,'w') as hdf:

        # patching for one frame
        if method =='rec_circ_patch':
            dset_waxs[0] = rec_circ_patch_one_frame(dset_waxs[0], args)     # here args is patches
        elif method =='thresholding':
            _, _, dset_waxs[0] = threshold_patch_one_frame(dset_waxs[0], args=args)
        elif method =='thr_rec_circ_patch':
            dset_waxs[0] = threshold_rec_circ_patch_one_frame(dset_waxs[0], args)
        else:
            raise Exception("something went wrong")
        
        # save patched image
        dset = hdf.create_group(f'/{h5_top_group(masked_file)}/primary/data')
        dset.create_dataset('pil1M_image', data = dset_saxs_no_mask, compression="lzf")   # no change 
        dset.create_dataset('pilW2_image', data = dset_waxs, compression="lzf")           # patched waxs

    ### Lin Yang's Code for 1-D averaging        
    azimuthal_averaging(masked_file, qgrid, n_proc=1, exp_folder = exp_folder)

    ### extract axes parameter   
    f, axs = plt.subplots(1,2, figsize = (12,6), num=f'{file}') if axes==None else axes
    
    ### Plot image after patches
    img = np.clip(dset_waxs[0], valid_range_min, valid_range_max, dtype = np.int8)
    im = axs[0].imshow(img, cmap = 'rainbow')     # remember each pixel value is limited by clipped value [0,10]
    # show_colorbar(im,f,axs[0])
    axs[0].set_title(f'Frame = {frame}')

    ### get WAXS data after patching
    Iq_M = read_Iq(masked_file, scattering)[0]      # read_Iq(masked_file, scattering) shape is (1, 690) dataset has only one frame masked frame file (temorary will be deleted after)
    Iq_S = read_Iq(file, scattering, frame)         # here dataset has only many frame - no masking file
    axs[1].plot(qgrid, Iq_S, color='#1f77b4', label= f'No Mask - {frame}')   
    axs[1].plot(qgrid, Iq_M, color='orange', label= f'Masked - {frame}')
    axs[1].set(xlabel = "$q (\AA^{-1})$", ylabel = "$I$", xscale='linear', yscale = 'linear' );
    axs[1].legend()
    ### axs[1].errorbar(qgrid2, dset_merged[0][0], dset_merged[0][1], label=f'{scattering} {frame}')   # here dataset has only one frame

    plt.suptitle(f'{file}')
    plt.tight_layout()  #    plt.subplot_tool() 

    # delete temporary file
    os.remove(masked_file)

    return Iq_M

### circular averaging after patching
def circ_avg_from_patches(source_file, qgrid, args, method = 'rec_circ_patch', exp_folder = ""):
    """
    rectangular/circular patch:
        masked_file = circ_avg_from_patches(source_file, qgrid, args=patches)
    thresholding patch:
        args = a_min, a_max, thr   # args is tuple ; different from previous one --> dset_waxs_sum will be calculated here
        masked_file = circ_avg_from_patches(source_file, qgrid, args = args, method = 'thresholding')
    """
    ## specs
    masked_file = f'{h5_top_group(source_file)}_masked.h5'

    ## computation
    if os.path.isfile(masked_file):
        print(f'DID not PATCH {masked_file} already exists - did not copy, ')
        print(f'1-D Averaging is NOT performed as {masked_file} already exist ')
        print(f'TO do patching on {masked_file} delete the file first manually and run this cell again')


    else:
        print(f'{masked_file} is being created in {os.getcwd()} ...')
        shutil.copy2(source_file, masked_file)       # create a new file
        print(f'{masked_file} copy is done')

        # read _WAXS2 source data, replace masked data and save masked out data on _WAXS2
        with h5py.File(masked_file,'r+') as hdf:
            try:
                dset = hdf.get(f'{h5_top_group(masked_file)}')
                del dset['processed']
                #del dset["_SAXS"], dset["_WAXS2"], dset["merged"]
            except:
                pass

            tic = time.perf_counter();     
            print(f'{masked_file} Loading data into a numpy array started ')
            dset = dset.get(f'/{h5_top_group(masked_file)}/primary/data')
            dset_waxs = np.array(dset.get('pilW2_image'))        # np.array(dset.get('pilW2_image')), dset.get('pilW2_image')[...] 
            del dset['pilW2_image']
            print(f'{masked_file} Loading data into a numpy array finished in {time.perf_counter()-tic} seconds')


            # Loop over the data to mask out the patches
            tic = time.perf_counter();     
            print(f'{masked_file} Patching Started ')
            dset_waxs_sum = np.sum(dset_waxs,axis=0) if method =='thresholding' else None   # summing all the frame values for thresholding
            for frame in range(len(dset_waxs)):
                if method =='rec_circ_patch':
                    dset_waxs[frame] = rec_circ_patch_one_frame(dset_waxs[frame], args)     # here args is patches
                elif method =='thresholding':
                    a_min, a_max, thr = tuple(args)   # making sure args is tuple
                    _, _, dset_waxs[frame] = threshold_patch_one_frame(dset_waxs[frame], (dset_waxs_sum, a_min, a_max, thr) )  # here args = (a_min, a_max, thr)
                elif method =='thr_rec_circ_patch':
                    dset_waxs[frame] = threshold_rec_circ_patch_one_frame(dset_waxs[frame], args)
                else:
                    raise Exception("Patching failed")
            print(f'{masked_file} Patching finished in {time.perf_counter()-tic} seconds')


            tic = time.perf_counter();     
            print(f'{masked_file} patched pilW2_image dataset creation staring ... ')
            # 'lzf' (no compression_opts), chunks=(1,1043,981), compression_opts=1
            dset.create_dataset('pilW2_image', data = dset_waxs, compression="gzip", compression_opts=1)  
            print(f'{masked_file} patched pilW2_image dataset creation finished in {time.perf_counter()-tic} seconds')


        ## create 1d data from lin Yang's code (py4xs packages are used here)
        azimuthal_averaging(masked_file, qgrid, n_proc=8, exp_folder = exp_folder)

        ## setting patch attributes on the processed folder
        set_patch_attributes(masked_file, args, method)

        return masked_file

def saxs_diff_image(file, frame, f, ax):
    """
        saxs_diff_image(file, frame, f, ax)
    """
    ### Read SAXS, WAXS Diffraction patterns
    valid_range_min, valid_range_max = (-1,10)      # just for SAXS/WAXS Display purpose

    with h5py.File(file,'r') as hdf:
        dset = hdf.get(f'{h5_top_group(file)}/primary/data')
        
        dset_saxs = dset['pil1M_image'][frame]
        dset_saxs = np.clip(dset_saxs, valid_range_min, valid_range_max, dtype = np.int8)
        im = ax.imshow(dset_saxs, cmap=plt.cm.rainbow); show_colorbar(im,f,ax)
        ax.set_title(f'SAXS Frame = {frame}', fontsize=8)

def waxs_diff_image(file, frame, f, ax):
    """
        waxs_diff_image(file, frame, f, ax)
    """    
    ### Read SAXS, WAXS Diffraction patterns
    valid_range_min, valid_range_max = (-1,10)      # just for SAXS/WAXS Display purpose

    with h5py.File(file,'r') as hdf:      
        dset = hdf.get(f'{h5_top_group(file)}/primary/data')
        dset_waxs = dset['pilW2_image'][frame]
        dset_waxs = np.clip(dset_waxs, valid_range_min, valid_range_max, dtype = np.int8)
        im = ax.imshow(dset_waxs, cmap=plt.cm.rainbow); show_colorbar(im,f,ax)
        ax.set_title(f'WAXS Frame = {frame}', fontsize=8)

def extract_line_indices(Tsplits, Tpoints, Startidx, RList=[]):
    """
    function call: indices = extract_line_indices(3, 109, 2)   # 3 line fits, maximum indices(excluding), starting indices
    """
    min_points = 2
    if Tsplits ==2:
        # Line1-Startidx, Line1-Startidx+increase points, Line2 Start idx =  Line1-Startidx+increase points, last point
        return [ RList + [[Startidx, Startidx+Npoints], [Startidx+Npoints, Tpoints]] \
                         for Npoints in range(min_points,Tpoints) \
                             if (Tpoints- (Startidx+Npoints))>=(Tsplits-1)*min_points]  # total points - (starting idx + last point) >= min_point to make sure we can fit line
    else:
        return [extract_line_indices(Tsplits-1, Tpoints, Startidx+Npoints, RList=RList+[[Startidx, Startidx+Npoints]]) \
                         for Npoints in range(min_points,Tpoints) \
                             if (Tpoints- (Startidx+Npoints))>=(Tsplits-1)*min_points]

# unequal level of list depth/nesting
def flatten(S):
    """
    l = [2,[[1,2]],1]
    list(flatten(l))
    """
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    return S[:1] + flatten(S[1:])

def linear_fit(xData, yData, ):
    """
    function call:  slope, intercept, rsq, p_val, std_err = linear_fit(xData, yData)
    """
    slope, intercept, r_val, p_val, std_err = stats.linregress(xData, yData)  # slope, intercept, r_val, p_val, std_err
    rsq = r_val**2;   # rsquare value    
    return np.round([slope, intercept, rsq, p_val, std_err], 4)

def plot_linear_lines(xData, yData, indices, args):
    """
    indices = [2,17,17,52,52,109]
    function call:  plot_linear_lines(xData, yData, indices)
    """
    f,ax = args
    Nsplits = int(len(indices)/2)                                   # retriving number of splits
                                                   
    color = ['red', 'black', 'green', 'blue', 'brown', 'magenta','cyan','purple']   # color sets for 3 line fittings only

    for i in range(0,Nsplits):
        idx = range(indices[i*2],indices[i*2+1])    # create indices range for xData and yData
        X  = xData[idx]            # qgrid low value
        y  = yData[idx]            # qgrid low value
        slope, intercept, _, _, _ = linear_fit(X, y)
        yfit  = np.polyval([slope, intercept], X)

        Rxc = np.sqrt(-slope*2)        # determine redius of cross sectional gyration
        ax.scatter(X, y, color=color[2*i],   label = f'exp data{i+1}')                       # scatter plot
        ax.plot(X, yfit, color=color[2*i+1], label = f'Rxc_{i+1}={Rxc:0.3f} I(0)_{i+1}={intercept:0.3f}', linewidth=3)              # polynomial fitting

    ax.set(xlabel = 'Q^2' , ylabel = 'log(Iq*Q)', xscale='linear', yscale = 'linear', title = f'Rg and I(0)' )
    ax.legend()

def optimize_best_lines(IqBS, qgrid, Nsplits, LastIdx, StartIdx, print_summary=False, show_plot=False, save_csv = False,):
    """
        This function operates on each frame
        df = optimize_best_lines(IqBS[frame], qgrid2, Nsplits, LastIdx, StartIdx, print_summary=True, show_plot=True)
    """
    # semi spces
    IqQ = np.log(IqBS*qgrid)                            # background subtracted Iq
    QQ  = np.square(qgrid)                                     # squaring q values
    xData = QQ
    yData = IqQ

    # computations
    indices = extract_line_indices(Nsplits, LastIdx, StartIdx)         # no of lines want to fit, maximum range (excluding), starting idx
    #print(indices)
    result = np.array(flatten(indices)).reshape(-1,Nsplits*2)          # create at 2D matrix of combinations 
    #print(result)                                                     # print total combinations matrix
    df = pd.DataFrame(result)                                          # create dataframe for the rest of the computations 

    columns = []
    for i in range(0,Nsplits):
        for cols in ['Tpoints-', 'qgridL-', 'qgridH-', 'slope-', 'Rxc-', 'I(0)-', 'rsq-', 'std_err-', 'std_err-']:
            df[cols+f'{i+1}'] = np.nan

    # column operations
    for i in range(0,Nsplits):    # use tqdm(range(0,Nsplits)) if needed be
        LL = i*2
        HL = LL+1
        df[f'Tpoints-{i+1}'] = df[HL] - df[LL]   # i starting from 1
        df[f'qgridL-{i+1}']  = qgrid[df[LL]]    # qgrid low value
        df[f'qgridH-{i+1}']  = qgrid[df[HL]]    # qgrid high value

        # row operations - linear reagression plotting
        for idx,(j,k) in enumerate(zip(df[LL].values, df[HL].values)):
            df.loc[idx,f'slope-{i+1}'], df.loc[idx, f'I(0)-{i+1}'], df.loc[idx, f'rsq-{i+1}'], _, df.loc[idx, f'std_err-{i+1}'] = linear_fit(xData[j:k], yData[j:k], )   # slope, intercept, rsq, p_val, std_err
            df.loc[idx,f'Rxc-{i+1}'] = np.round(np.sqrt(-df.loc[idx,f'slope-{i+1}']*2),4)

    # Summing up all rsq = rsq1+ rsq2 + rsq3
    df['rsq'] = 0; 
    for i in range(0,Nsplits):
        df['rsq'] += df[f'rsq-{i+1}']
   
    if save_csv == True:
        df.to_csv('output.csv')                                                   # create output.csv file
    
    try:       #Code that may raise an error
        np.isnan(np.sum(df.iloc[df['rsq'].idxmax()]))                                 # plotting/summary is possible if rsq doesn't have all nan values means no idxmax
    except:    #code to run if error occurs
        print('No best indices to plot - ending here')   

    else:      #code to run if no error is raised        
        best_indices = np.array(df.iloc[df['rsq'].idxmax()][0:Nsplits*2],dtype=int)    # find best indices for plotting 0:Nsplits*2 --> Nsplits 3 = column indices 0 1 2 3 4 5
        
        if print_summary == True:
            print(f'Total Combinations = {len(result)}\n')
            print('Summary of results Rsq -- \n',df.iloc[df['rsq'].idxmax()])         # print the location where rsq maximum
            pd.set_option('display.max_columns',None)                                 # display all dataframe columns
            print('More Summary of results Rsq -- \n',df[df['rsq'] > Nsplits-0.25])   # summary when Rsq > (3-0.15)=2.85
            ax = df['rsq'].plot.hist(bins=20, title='Rsq Histogram')                   # the distribution of rsq 
            ax.set_xlabel('rsq -->')

        if show_plot == True:
            plot_linear_lines(xData, yData, best_indices)

    #print(df.head())
    return df

### make lists of files for multiprocessing
def multiprocessing_lists_of_files_list(seek_str = '.h5', category_max_size_GB = 30):
    '''
        multiprocessing_lists_of_files_list(seek_str = '.h5', category_max_size_GB = 30)
    '''

    files_name_by_size = sorted(cwd_files_search_with(seek_str), key= lambda x: os.stat(x).st_size, reverse=True)
    files_size_by_size = [round(os.stat(file_name).st_size /1024/1024/1024, 2) for file_name in files_name_by_size] 
    df = pd.DataFrame({'Name':files_name_by_size , 'Size': files_size_by_size, 'Category': '' })


    sizes = []
    count = 1
    for idx, size_ in enumerate(df.Size):
        sizes.append(size_)   # append sizes list
        if sum(sizes) < category_max_size_GB:   
            df['Category'][idx] = f'group-{count}'   # write category
        else:
            sizes = [sizes.pop(-1)]                  # sizes list updated to the last size when more than 30 GB
            count +=1
            df['Category'][idx] = f'group-{count}'   # more than 30GB new category 
            
    print(df)

    processes_list = []
    for i in range(count):
        processes_list.append(list(df['Name'][df['Category'] == f'group-{i+1}']))
    #print(processes_list)

    return processes_list

# returns list of selected colormap
def cmap_list():
    selected_cmap = [
            'binary', 'gist_yarg', 'gist_gray', 'gray','winter', 'cool', 'hot', 'gist_heat', 'copper',
            'Spectral', 'coolwarm', 'bwr', 'seismic',
            'Accent', 'Set2', 'Set3', 'tab10', 'tab20c',
            'brg','gist_rainbow', 'rainbow', 'jet', 'turbo','gist_ncar', 'cividis', 'viridis']
    return selected_cmap


### func to interpolate Iq values
def interpolate_missing(A):
    # indx, A = interpolate_missing(Iq_M_WAXS[1])

    # https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    ok = ~ np.isnan(A)
    xp = ok.ravel().nonzero()[0]
    fp = A[~ np.isnan(A)]
    x  = np.isnan(A).ravel().nonzero()[0]
    A[np.isnan(A)] = np.interp(x, xp, fp)

    return x, A
