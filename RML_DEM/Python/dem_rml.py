import numpy as np
import pandas as pd
import os
import copy
import sys


def ReadAIAErrorTable(error_table_folder):
    """
    
    Function for reading an AIA error table, needed to compute the uncertainty on the AIA data
    
    
    Parameters
        ----------
        error_table_folder: string
                        path of the folder containing 'aia_V2_error_table.txt'. This error table is taken from
                        the SSW folder '/ssw/sdo/aia/response/'
    
    Returns
        -------
        wav: numpy array
            array containing the wavelength values of the different AIA channel
        
        dnperpht: numpy array
            array containing the covrsion factor between DN and photons for each AIA channel  
        
        compressratio: numpy array
            array containing the on-board compression factors for each AIA channel 
        
        chiantierr: numpy array
            array containing the systematic CHIANTI errors in each AIA channel 
        
        eveerr: numpy array
            array containing the systematic error due to normalizing the effective area using 
            cross-calibration with SDO/EVE
        
        calerr: numpy array
            array containing the systematic error due to uncertainty in the preflight photometric calibration 
            for each AIA channel 
            
    """
    
    txt_path = os.path.join(error_table_folder, 'aia_V2_error_table.txt')
    df = pd.read_csv(txt_path, sep='[ ]+|\t')
    
    wav           = np.array(df['WAVELNTH'])
    dnperpht      = np.array(df['DNPERPHT'])
    compressratio = np.array(df['COMPRESS'])
    chiantierr    = np.array(df['CHIANTI'])
    eveerr        = np.array(df['EVEERR'])
    calerr        = np.array(df['CALERR'])
    
    return wav, dnperpht, compressratio, chiantierr, eveerr, calerr

def AIAEstimateError(counts, sorted_wavs, n_avg=1, temperature=False, evenorm=False, cal=False):
    """
    
    Function for computing the uncertainty on experimental AIA data (based on the IDL routine 
    'aia_bp_estimate_error.pro')
    
    
    Parameters
        ----------
    
        counts: numpy array 
            array containing the AIA data [DN]. The shape is N_CHANNELS (x DIM_X x DIM_Y)
    
        sorted_wavs: numpy array 
            array containing the wavelength values ordered as the channels in counts 
            (e.g., [94, 131, 171, 193, 211]).
    
        n_avg: int
            an integer specifying how many measurements (adjacent pixels, or consecutive images) were averaged 
            to produce the measured counts. Default, 1 (i.e. no averaging).
    
        temperature: bool
            if set, the systematic error due to uncertainty in the CHIANTI data used to generate the 
            temperature response function is included. Default, False
    
        evenorm: bool
            if set, the systematic error due to normalizing the effective area using cross-calibration 
            with SDO/EVE is included. Default, False
            
        cal: bool
            if set, then the systematic error due to uncertainty in the preflight photometric calibration is included 
            (ignored if 'evenorm' is set). Default, False
    
    
    Returns
        -------
        sigmadn: numpy array 
            array containing the uncertainty on AIA data [DN]. The shape is N_CHANNELS (x DIM_X x DIM_Y)
    
    """
    
    # Read AIA error table
    error_table_folder = './tables/'

    wav, dnperpht, compressratio, chiantierr, eveerr, calerr_tab = ReadAIAErrorTable(error_table_folder)

    idx = []
    for i in range(len(sorted_wavs)):
        this_idx = np.where(wav == sorted_wavs[i])
        idx.append(this_idx[0][0])

    idx           = np.array(idx)
    wav           = wav[idx]
    dnperpht      = dnperpht[idx]
    compressratio = compressratio[idx]
    chiantierr    = chiantierr[idx]
    eveerr        = eveerr[idx]
    calerr_tab    = calerr_tab[idx]
    
    #************* Compute errors
    
    dim = counts.shape #list(counts.shape)
    if len(dim) > 0:
        dim0 = dim[0]
        dim1 = np.prod(np.array(dim[1:]))
        counts = np.reshape(counts, (dim0,dim1))
    else:
        counts = np.expand_dims(counts, axis=1)

    
    # Shot noise
    shotnoise = np.zeros(counts.shape)
    for i in range(dim[0]):
        
        # Short version of:
        
        # numphot   = counts[i,:] / dnperpht[i]
        # stdevphot = np.sqrt(numphot)
        # shotnoise[i,:] = stdevphot[i,:] * dnperpht[i]
        # shotnoise[i,:] = shotnoise[i,:] / np.sqrt(n_avg)
        
        shotnoise[i,:] = np.sqrt(counts[i,:] * dnperpht[i]/n_avg) 
    
    
    # Dark noise
    darknoise = 0.18
    
    # Read noise
    sumfactor = np.sqrt(n_avg) / n_avg
    readnoise = 1.15 * sumfactor
    
    # Quantum noise
    quantnoise = 0.288819 * sumfactor
    
    # Compression noise
    compressnoise = np.zeros(counts.shape)
    for i in range(dim[0]):
        compressnoise[i,:] = shotnoise[i,:] / compressratio[i]
     
    compressnoise[np.where(compressnoise < 0.288819)] = 0.288819
    idx_lowcounts = np.where(counts < 25)
    if len(idx_lowcounts[0]) > 0:
        compressnoise[idx_lowcounts] = 0

    compressnoise = compressnoise * sumfactor
    
    # Chianti noise
    chiantinoise = np.zeros(counts.shape)
    if temperature:
        
        for i in range(dim[0]):
            chiantinoise[i,:] = counts[i,:] * chiantierr[i]
    
    # Calibration noise
    calerr = np.zeros(dim[0])
    if evenorm:
        calerr = eveerr
    elif cal:
        calerr = calerr_tab
        
    calibnoise = np.zeros(counts.shape)
    for i in range(dim[0]):
            calibnoise[i,:] = counts[i,:] * calerr[i]
    
    sigmadn = np.sqrt(shotnoise**2 + darknoise**2 + readnoise**2 + quantnoise**2 + compressnoise**2 + \
                      chiantinoise**2 + calibnoise**2)
    
    sigmadn = np.reshape(sigmadn, dim)
    
    return sigmadn


def dem_rml(aia_data, aia_data_err, exptime, tresp, logT, dlogT, start_lam=1, tol=1e-3, maxiter=100, \
            n_lam=30, lam_factor=1.5, n_real=25, uncertainty=False, silent=False, lam_pixel=None):
    """
    
    Function for reconstructing Differential Emission Measure (DEM) profiles from SDO/AIA data by means of a Regularized
    Maximum Likelihood (RML) method (Massa and Emslie, A&A, 2023).
    
    Parameters
        ----------
        
        aia_data: numpy array
            array containing the AIA data [DN/s]. The shape is N_CHANNELS (x DIM_X x DIM_Y)
                
        aia_data_err: numpy array 
            array containing the uncertainty on AIA data [DN/s]. The shape is N_CHANNELS (x DIM_X x DIM_Y)
    
        exptime: numpy array
            array containing the exposure time value for each AIA channel [s]. The shape is N_CHANNELS
            
        tresp: numpy array
            array containing the temperature response function for each AIA channel [DN cm^5 pixel^-1 s-1]]. 
            The shape is N_CHANNELS x N_TEMPERATURES
            
        logT: numpy array
            array containing the value of the base 10 logarithm of the center of each temparture bin. 
            The shape is N_TEMPERATURES
        
        dlogT: numpy array
            array containing the value of the width of the base 10 logarithm of the temperature bins.
            The shape is N_TEMPERATURES
            
        start_lam: float
            Initial value of the regularization parameter (to be used as the starting point for the Morozov's 
            discrepancy principle). Default, 1.
            
        tol: float
            tolerance value to be used in the stopping criterion (|| x_k+1 - x_k || < tol * || x_k ||). Default, 1e-3
            
        maxiter: int
            maximum number of iterations (for each lambda value). Default, 100
            
        n_lam: int
            number of values of the regularization parameter to be considered for the Morozov's discrepancy principle. 
            Default, 30
    
        lam_factor: float
            factor to be used for decreasing the value of the regularization parameter at each step of the Morozov's 
            discrepancy principle. The new value is computed by dividing the old value by lam_factor. Default, 1.5
            
        n_real: int
            number of times the data are perturbed with Poisson noise for computing the uncertainty on the reconstruction.
            Default, 25
            
        uncertainty: bool
            if True, uncertainty on the reconstruction is performed by means of the "confidence strip" approach. 
            Default, False
        
        silent: bool
            if True, no text is printed. Default, False
            
        lam_pixel: input, numpy array
            array containing the regularization parameter value to be considered for each AIA pixel. Default, None (i.e., the
            regularization parameter value is estimated by means of the Morozov's discrepancy principle.
            
    Returns
        -------
        dem: numpy array 
            array containing the values of the reconstructed DEM profiles. The shape is N_TEMPERATURES (x DIM_X x DIM_Y)
            
        dem_error: numpy array
            array containing the values of the uncertainty on the reconstructed DEM profiles. 
            The shape is N_TEMPERATURES (x DIM_X x DIM_Y)
            
        lam_pixel: numpy array
            array containing the values of the reularization parameter adopted for each AIA pixel. 
            The shape is DIM_X x DIM_Y
    """
    
    #******************** DEM reconstruction
    
    data_dim  = aia_data.shape
    n_temp    = len(logT) # number of temperature bins
    
    if len(data_dim) == 1:
        xx    = np.zeros((n_temp,1)) + 1
        y     = np.expand_dims(aia_data, axis=1)
        err_y = np.expand_dims(aia_data_err, axis=1)
        # Array defining the regularization term in RML
        reg   = np.expand_dims(10**(2*logT)*np.log(10**dlogT)/1e32, axis=1)
        n_pix = 1
        if lam_pixel is not None:
            this_lam = lam_pixel
            if len(lam_pixel.shape) == 0:
                lam_pixel = np.zeros(n_pix) + lam_pixel
            
        
    elif len(data_dim) == 2:
        n_pix  = data_dim[1]
        xx     = np.zeros((n_temp, n_pix)) + 1
        y      = aia_data
        err_y  = aia_data_err
        # Array defining the regularization term in RML
        reg    = 10**(2*logT)*np.log(10**dlogT)/1e32
        reg    = np.repeat(np.expand_dims(reg, axis=1), n_pix, axis=1)
        if lam_pixel is not None:
            this_lam = np.repeat(np.expand_dims(lam_pixel, axis=0), n_temp, axis=0)
        
        
    elif len(data_dim) == 3:
        n_pix  = data_dim[1]*data_dim[2]
        xx     = np.zeros((n_temp, n_pix)) + 1
        y      = np.reshape(aia_data, (data_dim[0],n_pix))
        err_y  = np.reshape(aia_data_err, (data_dim[0],n_pix))
        # Array defining the regularization term in RML
        reg    = 10**(2*logT)*np.log(10**dlogT)/1e32
        reg    = np.repeat(np.expand_dims(reg, axis=1), n_pix, axis=1)
        if lam_pixel is not None:
            lam_pixel = np.reshape(lam_pixel, (n_pix,))
            this_lam  = np.repeat(np.expand_dims(lam_pixel, axis=0), n_temp, axis=0)
        
    else:
        print('Error: aia_data -> invalid number of dimensions', file=sys.stderr)
    
    #To be used for the confidence strip approach
    if uncertainty:
        
        y_dn = np.zeros(y.shape) 
        
        for i in range(data_dim[0]):
            
            y_dn[i,:] = y[i,:] * exptime[i] # Makes the data in DN unit, so that we can apply Poisson perturbation
        
       
    # Change name of the temperature response function (just a convention)
    H = tresp
    
    # Initialize the solution.
    dim_xx = xx.shape
    xx_sol = np.zeros(dim_xx) 
    xx     = np.zeros(dim_xx) + 1 #solution for the current value of lambda
    
    if not silent:
            print()
            print("Start")
            print()    
    
    if lam_pixel is not None:
        
        # RML iterative method (for the current value of the regularization parameter)
        Ht1 = np.transpose(H) @ (y*0 + 1) + this_lam * reg

        for i in range(maxiter):

            xx_old = copy.deepcopy(xx)

            Hx = H @ xx

            z = np.divide(y, Hx, out=np.zeros_like(y), where= Hx!=0)

            Hz = np.transpose(H) @ z

            update = np.divide(Hz, Ht1, out=np.zeros_like(Hz), where= Ht1!=0)

            xx = xx * update


            # Check which pixels satisfy the stopping criterion
            iidx =  np.where(np.sqrt(np.sum((xx_old - xx)**2,axis=0)) <= tol * np.sqrt(np.sum(xx_old**2,axis=0)))
            iidx = iidx[0]

            if (i > 0) and (len(iidx) > 0):

                if len(dim_xx) > 0:
                    xx[:,iidx] = xx_old[:,iidx]
                else:
                    xx = copy.deepcopy(xx_old)
                    
        xx_sol = xx
        
    else:
        
        # Indices  of the pixels for which the solution has to be computed 
        idx = np.indices((n_pix,))
        idx = idx[0,:]

        # Initialize the value of the regularization parameter
        this_lam = start_lam

        # Initialize matrix containing the selected value of the regularization parameter for each pixel
        # To be used for perfoming estimate of the error in each reconstruction
        lam_pixel = np.zeros(n_pix)

        for j in range(n_lam):

            if not silent:
                print("Current value of the regularizarion parameter: %.3f" % this_lam)

            # RML iterative method (for the current value of the regularization parameter)
            Ht1 = np.transpose(H) @ (y*0 + 1) + this_lam * reg[:,idx]

            for i in range(maxiter):

                xx_old = copy.deepcopy(xx)

                Hx = H @ xx

                z = np.divide(y, Hx, out=np.zeros_like(y), where= Hx!=0)

                Hz = np.transpose(H) @ z

                update = np.divide(Hz, Ht1, out=np.zeros_like(Hz), where= Ht1!=0)

                xx = xx * update


                # Check which pixels satisfy the stopping criterion
                iidx =  np.where(np.sqrt(np.sum((xx_old - xx)**2,axis=0)) <= tol * np.sqrt(np.sum(xx_old**2,axis=0)))
                iidx = iidx[0]

                if (i > 0) and (len(iidx) > 0):

                    if len(dim_xx) > 0:
                        xx[:,iidx] = xx_old[:,iidx]
                    else:
                        xx = copy.deepcopy(xx_old)


            # Compute reduced chi2
            y_pred   = H @ xx        
            chi2     = np.sum( (y_pred-y)**2/err_y**2, axis=0) / (data_dim[0]-1)


            # Check which pixels satisfy the Morozov discrepancy principle
            this_idx_gt = np.where(chi2 > 1) #n_chi_gt
            this_idx_gt = this_idx_gt[0]
            this_idx_le = np.where(chi2 <= 1) #, n_chi_le
            this_idx_le = this_idx_le[0]


            # If a pixel satifies the Morozov discrepancy principle, then the corresponding solution is 
            # saved in 'xx_sol' and it will not be computed for other values of the regularization parameter
            if len(this_idx_le) > 0:

                idx_le   = idx[this_idx_le]
                idx      = idx[this_idx_gt]

                xx_sol[:,idx_le] = xx[:,this_idx_le]
                y  = y[:,this_idx_gt]

                err_y = err_y[:,this_idx_gt]

                # save value of the regularization parameter
                lam_pixel[idx_le] = this_lam


            if len(this_idx_gt) > 0:

                if j < n_lam-1:
                    xx = xx[:,this_idx_gt] * 0 + 1
                else:
                    xx = xx[:,this_idx_gt]
            else:
                break

            # Compute new value of the regularization parameter

            this_lam = this_lam/lam_factor

            #If the solution corresponding to a specific pixel does not satisfy the Morozov discrepancy
            # for any value of the regularization parameter, then we compute it the starting value 
            # (we keep the most regularized solution)
            if (len(this_idx_gt) > 0) and (j == n_lam-2):

                this_lam = start_lam
                lam_pixel[idx] = this_lam

        if len(this_idx_gt) > 0:
            xx_sol[:,idx] = xx
        
    if not silent:
        print()
        print("End")
        print()
        print()
        
    #******************** Uncertainty estimation
    
    if uncertainty:
        
        if not silent:
            print("Estimate of the error: confidence strip approach")
            print()
        
        lam_pixel_unc = np.repeat(np.expand_dims(lam_pixel,0),n_temp,axis=0)    #np.repeat(, (n_temp,1))#transpose()

        if len(dim_xx) > 1: 
            xx_error = np.zeros((dim_xx[0], dim_xx[1], n_real))
        else:
            xx_error = np.zeros((dim_xx, n_real))
        
        
        for jj in range(n_real):
            
            if not silent:
                print("Realization n " + str(jj+1))
            
            # Perturb data
            np.random.seed(jj)
            this_y  = np.random.poisson(y_dn).astype('float32')

            for i in range(data_dim[0]):
                this_y[i,:] = this_y[i,:] /exptime[i]

            
            xx  = np.zeros(dim_xx) + 1. # solution for the current value of lambda
            Ht1 = np.transpose(H) @ (this_y*0.+1.) + lam_pixel * reg

            for i in range(maxiter):

                xx_old = xx

                Hx = H @ xx

                z = np.divide(this_y, Hx, out=np.zeros_like(this_y), where= Hx!=0)

                Hz = np.transpose(H) @ z

                update = np.divide(Hz, Ht1, out=np.zeros_like(Hz), where= Ht1!=0)

                xx = xx * update

                # Check which pixels satisfy the stopping criterion
                iidx =  np.where(np.sqrt(np.sum((xx_old - xx)**2,axis=0)) <= tol * np.sqrt(np.sum(xx_old**2,axis=0)))
                iidx = iidx[0]

                if (i > 0) and (len(iidx) > 0):

                    if len(dim_xx) > 0:
                        xx[:,iidx] = xx_old[:,iidx]
                    else:
                        xx = copy.deepcopy(xx_old)

            if len(dim_xx) > 0:
                xx_error[:,:,jj] = xx
            else:
                xx_error[:,jj] = xx

        
        if len(dim_xx) > 1: 
            xx_error = np.std(xx_error/1e20, axis=2)*1e20
        else:
            xx_error = np.std(xx_error/1e20, axis=1)*1e20


    if uncertainty:
        dem_error = xx_error
    else:
        dem_error = xx_sol * 0.
        
    #********************
    
    dem = xx_sol
    
    if len(data_dim) == 1:
        dem       = np.squeeze(dem)
        dem_error = np.squeeze(dem_error)
    elif len(data_dim) == 3:
        dem       = np.reshape(dem, (n_temp, data_dim[1], data_dim[2]))
        lam_pixel = np.reshape(lam_pixel, (data_dim[1], data_dim[2]))
        dem_error = np.reshape(dem_error, (n_temp, data_dim[1], data_dim[2]))
        
    return dem, dem_error, lam_pixel