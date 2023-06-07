;+
;
; NAME:
;
;   dem_rml
;
; PURPOSE:
;
;   Reconstruct the Differential Emission Measure (DEM) from AIA data by means of a 
;   Regularized Maximum Likelihood (RML) approach
;
; CALLING SEQUENCE:
;
;   dem = dem_rml(data_in, data_err, tresp, temp_bin_edges)
;
; INPUTS:
;   data_in: AIA data (DN s^-1). Dimensions are: N_CHANNELS x (NX x NY, if arrays or images), where 
;            N_CHANNELS is the number of AIA data considered (default 5, no 335 A) and NX x NY are the number of 
;            AIA pixels considered
;   
;   data_err: estimate of the data uncertainty. Same dimension as 'data_in'
; 
;   tresp: AIA temperature response function. Dimensions are  N_CHANNELS x N_TEMP, where N_TEMP is the number 
;          of temperature channels considered 
; 
;   temp_bin_edges: array containing the edges of the N_TEMP temperature bins considered. Dimension, N_TEMP+1
;
; OUTPUTS:
;
;   Structure containing:
;   
;   {dem:dem, dem_error:dem_error}
;   
;   - dem: Array of dimension N_TEMP x (NX x NY) containing the reconsruced DEM values for each temperature and pixel
;   - dem_error: Array of dimension N_TEMP x (NX x NY) containing an estimate of the uncertainty on the DEM reconstruction
;                performed by means of a "confidence strip" approach (if the keyword 'uncertainty' is set)
;
; KEYWORDS:
;
;   lam: starting value of the regularization parameter, default 1e-10
;   
;   tol: tolerance value used for the stopping criterion (|| x_k+1 - x_k || < tol * || x_k ||). Default, 
;   
;   maxiter: maximum number of iteration performed by RML (for a specific value of the regularization parameter). 
;            Default, 100
;            
;   n_lam: maximum number of values of regularization parameter that are considered. 
;          If the Morozov discrepancy principle is satified for each pixel before n_lam values are 
;          considered, then the method stops
;                  
;   lam_factor: factor which divides the starting value of the regularization parameter.
;               The j-th value of the regularization parameter is given by lam/lam_factor^k
; 
;   n_real: number of realizations used for proving an estimate of the uncertainty on the reconstructions by means of a
;           confidence strip approach
;           
;   uncertainty: if set, an estimate of the uncertainty on the DEM solution is performed by means of the "confidence 
;                strip" approach 
;           
; HISTORY: December 2022, Massa P., First version
;
; CONTACT:
;   paolo.massa@wku.edu
;-
function dem_rml, data_in, data_err, tresp, temp_bin_edges, $
                  lam=lam, tol=tol, maxiter=maxiter, $
                  n_lam=n_lam, lam_factor=lam_factor, exptime=exptime, n_real=n_real, uncertainty=uncertainty

  default, tol, 1e-3
  default, maxiter, 100
  default, lam, 1.
  default, n_lam, 30
  default, lam_factor, 1.5
  default, n_real, 25
  default, uncertainty, 1
  data_dim  = size(data_in, /dim)
  default, exptime, fltarr(data_dim[0]) + 1.
  
  if n_real le 0 then message, "n_real must be greater than 0."

  ;; Define xx (iterative solution, initilized with 1) and reshape y
  logT=get_edges(alog10(temp_bin_edges),/mean)
  dlogT=get_edges(alog10(temp_bin_edges),/width)
  n_temp    = n_elements(logT) ; number of temperature bins
  
  case n_elements(data_dim) of

    1: begin
      xx = fltarr(n_temp) + 1
      y  = float(data_in)
      err_y = float(data_err)
      ;; Array defining the regularization term in RML
      reg = 10d^(2.*logT)*alog(10d^dlogT)/1e32
      n_pix = 1
    end

    2: begin
      n_pix = data_dim[1]
      xx = fltarr(n_temp, n_pix) + 1
      y  = float(data_in)
      err_y = float(data_err)
      reg = cmreplicate(10d^(2.*logT)*alog(10d^dlogT)/1e32, n_pix)
    end

    3: begin
      n_pix = long(data_dim[1])*long(data_dim[2])
      xx = fltarr(n_temp, n_pix) + 1
      y  = float(reform(data_in, data_dim[0], n_pix))
      err_y = float(reform(data_err, data_dim[0], n_pix))
      reg = cmreplicate(10d^(2.*logT)*alog(10d^dlogT)/1e32, n_pix)
    end

    else: begin
      message, "Input data (DATA_IN) have an invalid number of dimensions."
    end

  endcase

  ; To be used for the confidence strip approach
  if keyword_set(uncertainty) then begin
    y_dn = y 
    for i=0,data_dim[0]-1 do y_dn[i,*] = y[i,*] * exptime[i] ; Makes the data in DN unit, so that we can apply 
                                                           ; Poisson perturbation
  endif
  
  ; Change name of the temperature response function (just a convention)
  H = tresp
  
  ; Indices  of the pixels for which the solution has to be computed 
  idx   = lindgen(n_pix)
  
  ; Initialize the value of the regularization parameter
  this_lam = lam
  
  ; Initialize the solution.
  dim_xx = size(xx, /dim)
  xx_sol = dblarr(dim_xx) 
  xx     = dblarr(dim_xx) + 1. ;; solution for the current value of lambda
  
  ; Initialize matrix containing the selected value of the regularization parameter for each pixel
  ; To be used for perfoming estimate of the error in each reconstruction
  lam_pixel = fltarr(n_pix)
  
  print
  print, "Start"
  print
  
  for j=0,n_lam-1 do begin
  
  print, "Current value of the regularizarion parameter: " + num2str(this_lam, format='(f9.4)')
  ;; RML iterative method (for the current value of the regularization parameter)
  Ht1 = H # (y*0.+1.) + this_lam * reg[*,idx]
  
  for i=1, maxiter do begin

    xx_old = xx

    Hx = transpose(H) # xx
    z = f_div(y , Hx)
    Hz = H # z
    update = f_div(Hz, Ht1)
    xx = xx * update
    
    ;; Check which pixels satisfy the stopping criterion
    iidx =  where(sqrt(total((xx_old - xx)^2.,1)) le tol * sqrt(total(xx_old^2.,1)), n_idx)
    if (i gt 1) and (n_idx gt 0) then begin

      if n_elements(dim_xx) gt 1 then begin
        xx[*,iidx] = xx_old[*,iidx]
      endif else begin
        xx = xx_old
      endelse

    endif

  endfor
  
  
  ;; Compute reduced chi2
  y_pred   = transpose(H) # xx
  chi2     = total( (y_pred-y)^2./err_y^2., 1) / (data_dim[0]-1.)
  ;; Check which pixels satisfy the Morozov discrepancy principle
  this_idx_gt = where(chi2 gt 1., n_chi_gt)
  this_idx_le = where(chi2 le 1., n_chi_le)
  
  ;; If a pixel satifies the Morozov discrepancy principle, then the corresponding solution is 
  ;; saved in 'xx_sol' and it will not be computed for other values of the regularization parameter
  if n_chi_le gt 0 then begin
  idx_le   = idx[this_idx_le]
  idx      = idx[this_idx_gt]
  
  xx_sol[*,idx_le] = xx[*,this_idx_le]
  y  = y[*,this_idx_gt]
  
  err_y = err_y[*,this_idx_gt]
  
  ; save value of the regularization parameter
  lam_pixel[idx_le] = this_lam
 
  endif
  
  if n_chi_gt gt 0 then begin
  if j lt n_lam-1 then begin
    xx = xx[*,this_idx_gt] * 0. + 1.
  endif else begin
    xx = xx[*,this_idx_gt]
  endelse
  endif else begin
    break
  endelse
  ;; Compute new value of the regularization parameter
  
  this_lam = this_lam/1.5
  
  ;; If the solution corresponding to a specific pixel does not satisfy the Morozov discrepancy
  ;; for any value of the regularization parameter, then we compute it the starting value 
  ;; (we keep the most regularized solution)
  if (n_chi_gt gt 0) and (j eq n_lam-2) then begin
    this_lam = lam
    lam_pixel[idx] = this_lam
  end
  endfor
  
  if n_chi_gt gt 0 then xx_sol[*,idx] = xx
  
  print
  print, "End"
  print
  print
  
  if uncertainty then begin
  print, "Estimate of the error: confidence strip approach"
  print
  
  lam_pixel = transpose(cmreplicate(lam_pixel, n_temp))
  
  
  
  if n_elements(dim_xx) gt 1 then begin
    xx_error = fltarr(dim_xx[0], dim_xx[1], n_real)
  endif else begin
    xx_error = fltarr(dim_xx, n_real)
  endelse
  
  for jj=0,n_real-1 do begin
    
    print, "Realization n " + num2str(jj+1)
    
    ;; Perturb data
    seed    = jj
    this_y  = poidev(y_dn,SEED = seed);randomn(seed, size(y_cs, /dim)) * err_y_cs
    
    for i=0,data_dim[0]-1 do this_y[i,*] = this_y[i,*] /exptime[i]
  
    xx     = dblarr(dim_xx) + 1. ;; solution for the current value of lambda
    
    Ht1 = H # (this_y*0.+1.) + lam_pixel * reg
    
    for i=1, maxiter do begin

      xx_old = xx

      Hx = transpose(H) # xx
      z = f_div(this_y , Hx)
      Hz = H # z
      update = f_div(Hz, Ht1)
      xx = xx * update

      ;; Check which pixels satisfy the stopping criterion
      iidx =  where(sqrt(total((xx_old - xx)^2.,1)) le tol * sqrt(total(xx_old^2.,1)), n_idx)
      if n_idx gt 0 then begin
      ;if (i gt 1) and (n_idx gt 0) then begin

        if n_elements(dim_xx) gt 1 then begin
          xx[*,iidx] = xx_old[*,iidx]
        endif else begin
          xx = xx_old
        endelse

      endif

    endfor
  
  
  if n_elements(dim_xx) gt 1 then begin
    xx_error[*,*,jj] = xx
  endif else begin
    xx_error[*,jj] = xx
  endelse
    
  endfor
  
  
  if n_elements(dim_xx) gt 1 then begin
    xx_error = stddev(xx_error/1e20, dim=3)*1e20
  endif else begin
    xx_error = stddev(xx_error/1e20, dim=2)*1e20
  endelse
  endif
  
  dem       = xx_sol
  if uncertainty then begin
    dem_error = xx_error
  endif else begin
    dem_error = xx_sol * 0.
  endelse
  if n_elements(data_dim) eq 3 then begin
    dem = reform(dem, n_temp, data_dim[1], data_dim[2])
    dem_error = reform(dem_error, n_temp, data_dim[1], data_dim[2])
  endif

  return, {dem:dem, dem_error:dem_error}

end