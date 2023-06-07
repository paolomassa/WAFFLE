;;***********************************************************************************************;;
;
;            Regularized Maximum Likelihood (RML) method for DEM inversion from AIA data
;
;;***********************************************************************************************;;

;; INSERT PATH OF THE FOLDER

folder = ""
add_path, folder + "/IDL/rml"

;;******************************** READ AIA FITS FILES ******************************************;;

folder_path = folder + "/aia_fits/"
data_name = "aia.lev1_euv_12s.2010-11-03T121518Z"

; Path of the AIA fits files
path_files = file_search(folder_path + data_name + "*")

; Extract wavelenght information from each file name
wav_array = strarr(n_elements(path_files))
data_name_len = strlen(data_name)
for i=0,n_elements(path_files)-1 do begin
  
  pos_data_name = strpos(path_files[i],data_name)
  pos_image     = strpos(path_files[i],".image")
  wav_array[i] = strmid(path_files[i], pos_data_name+data_name_len+1, $
                        pos_image-(pos_data_name+data_name_len+1))

endfor

; Sort fits file paths based on the wavelenghts
sorted_wav = ["94","131","171","193","211"]
sort_path_files = strarr(n_elements(sorted_wav))
for i=0,n_elements(sorted_wav)-1 do sort_path_files[i]=path_files[where(wav_array eq sorted_wav[i])]

; Read AIA maps and corresponding exposure time
aia_maps = []
exptime  = []
for i=0,n_elements(sort_path_files)-1 do begin
  fits2map,sort_path_files[i],aia_map
  aia_maps=[aia_maps,aia_map]

  read_sdo, sort_path_files[i], index
  exptime = [exptime, index.EXPTIME]

endfor

; Define matrix containing the AIA maps
aia_img = float(aia_maps.DATA)
; For avoiding numerical problems we substitute zero pixels with a low (but positive) value
idx = where(aia_img le 0, nzeros)
if nzeros gt 0 then aia_img[idx]=1e-3
; Normalize AIA images by exposure time (to make them in DN s^-1)
for i=0,n_elements(sorted_wav)-1 do aia_img[*,*,i] = aia_img[*,*,i]/exptime[i]

;;***************************** COMPUTE TEMPERATURE RESPONSE ************************************;;

;; TAKEN AND ADAPTED FROM https://github.com/ianan/demreg

; Define edges of the temperature bins to be considered
logtemps = 5.85-0.15/2. + findgen(13)*0.15
temps     = 10d^logtemps
; This is is the temperature bin mid-points
logt=get_edges(logtemps,/mean)
dlogT=get_edges(logtemps,/width)

; Compute AIA temperature response
date_obs   = aia_maps[0].TIME
tresp=aia_get_response(/temperature,/dn,/chianti,/noblend,/evenorm,timedepend_date=date_obs)

; Indices of the AIA channels to consider (no 335 A) 
idc=[0,1,2,3,4]

; Keep temperature values within the maximum and minimum edge defined in 'logtemps'
tr_logt=tresp.logte
gdt=where(tr_logt ge min(logtemps) and tr_logt le max(logtemps),ngd)
tr_logt=tr_logt[gdt]
TRmatrix=tresp.all[*,idc]
TRmatrix=TRmatrix[gdt,*]

; Avoid negative or zero values in the temperature response function
truse=dblarr(n_elements(TRmatrix[*,0]),n_elements(idc))
for i=0,n_elements(idc)-1 do begin
  goodtr=where(TRmatrix[*,i] gt 0.)
  badtr=where(TRmatrix[*,i] le 0.,nbadtr)
  truse[goodtr,i]=TRmatrix[goodtr,i]
  if (nbadtr gt 0) then truse[badtr,i]=min(TRmatrix[goodtr,i])
endfor

; Make the temperature response function per unit T
TR     = dblarr(n_elements(logT),n_elements(idc))
for i=0, n_elements(idc)-1 do TR[*,i]=interpol(truse[*,i], tr_logt, logT)*10d^logT*alog(10d^dlogT)

;;***************************** COMPUTE ERROR ON AIA DATA ************************************;;

dim = size(aia_img, /dim)
aia_error = fltarr(dim)
for i=0,dim[0]-1 do begin
for j=0,dim[1]-1 do begin

  aia_error[i,j,*] = aia_bp_estimate_error(reform(aia_img[i,j,*])*exptime, $
    [94,131,171,193,211], num_images=1)/exptime

endfor
endfor

;;********************************* DEM RECONSTRUCTIONS ************************************;;


;; Regularized Maximum likelihood (RML)
aia_img_transp   = transpose(aia_img,[2,0,1])
aia_error_transp = transpose(aia_error,[2,0,1])

demrml = dem_rml(aia_img_transp, aia_error_transp, TR, temps, exptime=exptime)
rml_dem       = demrml.dem
rml_dem_error = demrml.dem_error

;;************************************* PLOT RESULTS ***************************************;;

window,0,xsize=1000,ysize=800
cleanplot

!p.multi = [0,4,3]

charsize=2
dmin = [1e19,1e19,1e19,1e19,1e17,1e17,1e17,1e17,1e15,1e15,1e15,1e15]

loadct,5,/silent
for i=0,n_elements(logt)-1 do begin

  this_im = reform(rml_dem[i,*,*])
  dem_map = make_map(this_im,xc=aia_maps[0].xc, yc=aia_maps[0].yc, dx=aia_maps[0].dx, dy=aia_maps[0].dy)
  plot_map, dem_map, /log, charsize=charsize, dmin = dmin[i], dmax=3e22, $
    title="Log(T) = " + num2str(logt[i],format='(f9.2)')
endfor

;;************************************* SAVE RESULTS ***************************************;;

results_folder=folder + "/IDL/results/"
if ~file_exist(results_folder) then file_mkdir, results_folder
save, rml_dem, rml_dem_error, aia_img, aia_error, TR, filename=results_folder + "/results.sav" 

end
