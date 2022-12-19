;;***********************************************************************************************;;
;
;            Routine for plotting the DEM profiles obtained by means of REG and RML
;
;;***********************************************************************************************;;


;; INSERT PATH OF THE FOLDER

folder = ""

restore, folder + "/results/results.sav", /v 


;;************** SELECT PIXEL

;; The DEM corresponding to the selected pixel will be displayed
pix_x = 180
pix_y = 210

;;************** PLOT AIA IMAGES

;; The red cross corresponds to the selected pixel

wav = [94,131,171,193,211]

charsize=2.5
window,0,xsize=1300,ysize=400
cleanplot
!p.multi = [0,5,1]
for idx=0,n_elements(wav)-1 do begin

  this_im = reform(aia_img[*,*,idx])
  this_im[where(this_im eq 1e-3)] = 2
  aia_lct, wav=wav[idx], /load
  plot_image, alog(this_im), charsize=charsize, title="AIA " + num2str(wav[idx]), min=0.01, max=8.74
  linecolors
  oplot, [pix_x],[pix_y], psym=1, color=2, symsize=2, thick=2

endfor

;;************** DISPLAY DEM and FIT

this_dem_rml       = reform(rml_dem[*,pix_x, pix_y])
this_dem_rml_error = reform(rml_dem_error[*,pix_x, pix_y])

err_min_rml = this_dem_rml - this_dem_rml_error
err_min_rml = err_min_rml > 0.
err_max_rml = this_dem_rml + this_dem_rml_error


;; Data predicted from reconstructed DEM
y_pred_rml     = TR ## this_dem_rml

y             = aia_img[pix_x,pix_y,*]
y_error       = aia_error[pix_x,pix_y,*]

left  = [0.12, 0.42, 0.72, 0.12, 0.42, 0.72]
right = [0.35, 0.65, 0.96, 0.35, 0.65, 0.96]
low   = [0.08, 0.08, 0.08, 0.35, 0.35, 0.35]
up    = [0.29, 0.29, 0.29, 0.56, 0.56, 0.56]

thick=1.5
charsize=2.5
xticks=3
xtickv=[5.8,6.2,6.6,7.0]
symsize=2

window,2,xsize=800, ysize=800

set_viewport,0.12,0.96,0.65,0.9
temps=[0.5,1,1.5,2,3,4,6,8,11,14,19,25,32]*1d6
logtemps=alog10(temps)
mlogt=get_edges(logtemps,/mean)

loadct,5, /silent
plot, mlogt, this_dem_rml, title="DEM", charsize=charsize, xtit='Log!D10!N T', ytit='DEM [cm!U-5!N K!U-1!N]', /xst, /noe
linecolors
oplot, mlogt, this_dem_rml, color=2, thick=thick

errplot, mlogt, (err_min_rml > !y.crange[0]), (err_max_rml < !y.crange[1]), $
         width=0, thick=thick, color=2

ssw_legend, ["Chi2 RML: " + num2str(total((y_pred_rml-y)^2./y_error^2.)/4., format='(f9.2)')],$
  textcolors=[2,7], box=0, charsize=1.2, /top, /right



for i=0,n_elements(y)-1 do begin

  set_viewport,left[i],right[i],low[i],up[i]

  fov = 3*sqrt(y[i])
  plot, [0],[y[i]], psym=1, symsize=symsize, xrange=[-1,1], /yst, charsize=charsize, yrange=[y[i]-fov,y[i]+fov], $
    title="AIA " + num2str(wav[i]) + " A",XTICKFORMAT="(A1)", /noe, ytitle="[DN s!U-1!N]"
  errplot, [0],[y[i]-y_error[i]], [y[i]+y_error[i]]
  oplot, [0],[y_pred_rml[i]], color=2, psym=2, symsize=symsize, thick=thick

endfor

ssw_legend, ["Observed data", "Predicted from RML"], $
  textcolors=[255,2,7], box=0, right=1, top=1,charsize=1.2

!p.position = [0, 0, 0, 0]

end