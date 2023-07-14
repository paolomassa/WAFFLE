import drms
from drms import ServerConfig
from urllib.request import urlretrieve
from urllib.error import URLError, HTTPError
import os
from sunpy.map import Map
import datetime
from datetime import timedelta

from aiapy.calibrate.util import get_correction_table as get_correction_table
from aiapy.calibrate import normalize_exposure, register, update_pointing, correct_degradation
import time

from sunpy.coordinates import frames
import sunpy

import csv

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
from astropy.io.fits import CompImageHDU

import numpy as np

from torchvision.transforms import Resize
import torch

import glob

from paramiko import SSHClient
from scp import SCPClient
import dem_rml

import pytz

import cv2

import wget

import pandas as pd

import json

from dateutil import tz

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.colors as colors

#**********************************************************

def mkdir(a_dir):
    if(not os.path.exists(a_dir)):
        os.makedirs(a_dir)   

# Set server configuration
def configure_jsoc_server():
    server = ServerConfig(name="JSOC",
                          cgi_baseurl="http://jsoc2.stanford.edu/cgi-bin/ajax/",
                          cgi_show_series="show_series",
                          cgi_jsoc_info="jsoc_info",
                          cgi_jsoc_fetch="jsoc_fetch",
                          cgi_check_address="checkAddress.sh",
                          cgi_show_series_wrapper="showextseries",
                          show_series_wrapper_dbhost="hmidb2",
                          http_download_baseurl="http://jsoc2.stanford.edu/",
                          ftp_download_baseurl="ftp://pail.stanford.edu/export/")

    client = drms.Client(server=server,verbose=True)
    
    return client


def download_aia_data(wav, t_rec, segments, data_folder, timezone='US/Central'):
    
    # Create download folder
    full_disk_maps_folder = os.path.join(data_folder, t_rec[0].replace(":", ""))
    mkdir(full_disk_maps_folder)
        
    # Get fits file url
    website_url   = 'https://jsoc1.stanford.edu/'
    
    idx      = np.argsort(wav)
    wav      = wav[idx]
    t_rec    = t_rec[idx]
    segments = segments[idx]
    
    this_datetime_timezone = convert_utc_to_timezone(datetime.datetime.strptime(t_rec[0], '%Y-%m-%dT%H:%M:%SZ'), timezone=timezone)
    print("Start download AIA data recorded at " + this_datetime_timezone.strftime("%d-%m-%YT%H:%M:%S") + " " + timezone)
    
    aia_maps = []
    
    # Error will be true if it is not possible to download a file
    error = False
    for i in range(len(wav)):
    
        fits_file_url = website_url + segments[i]
       
        # Define fits file name
        filename = os.path.join(full_disk_maps_folder, 'aia_lev1_nrt2_'+ t_rec[i] + '_' + str(wav[i]) +'.fits').replace(":", "")

        # Download fits file
        try:
            urlretrieve(fits_file_url, filename)
        except (HTTPError, URLError):
            error = True
            continue
        
        aia_maps.append(Map(filename))
                
    print("Download completed!")
    return aia_maps, full_disk_maps_folder, error


def calibrate_full_disk_maps(aia_maps):
    
    calibrated_aia_maps = []
    
    for this_map in aia_maps:
        calibrated_aia_maps.append(register(this_map))
        
    return calibrated_aia_maps


def extract_submaps(aia_map, ar_lon, ar_lat, n_pix = 500):
    
    this_coord = SkyCoord(ar_lon*u.deg, ar_lat*u.deg, frame=frames.HeliographicStonyhurst)

    pix_x = aia_map.world_to_pixel(this_coord).x.value
    pix_y = aia_map.world_to_pixel(this_coord).y.value

    top_right   = aia_map.pixel_to_world((pix_x+n_pix//2-1)*u.pix,(pix_y+n_pix//2-1)*u.pix)
    bottom_left = aia_map.pixel_to_world((pix_x-n_pix//2)*u.pix,(pix_y-n_pix//2)*u.pix)
    
    submap = aia_map.submap(bottom_left, top_right=top_right)
    submap = Map(submap.data.astype(np.int16), submap.meta)
    
    return submap

def crop_full_disk_maps(aia_maps, ar_lon, ar_lat, arnum, dowloaded_data_folder, n_pix=500):
    
    cropped_maps_folder = dowloaded_data_folder + "_crop"
    mkdir(cropped_maps_folder)
    
    aia_submaps = []
    for aia_map in aia_maps:
        aia_submap = extract_submaps(aia_map, ar_lon, ar_lat, n_pix = n_pix)
        wav = aia_submap.meta['wavelnth']
        fitsname = os.path.join(cropped_maps_folder, 'aia_lev1_nrt2_' + str(wav) + '_ar' + str(arnum) + '.fits')
        sunpy.io.fits.write(fitsname, aia_submap.data, aia_submap.fits_header, overwrite=True) #, hdu_type=CompImageHDU
        aia_submaps.append(aia_submap)
        
    return aia_submaps


def crop_around_max_aia_94A(aia_submaps, npix_crop=200):
    
    submaps = []
    exptime = []
    
    aia_map_94 = Map(aia_submaps[0])
    
    # Find pixel corresponding to the maximum
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(aia_map_94.data)
    
    # Indices of the center
    xc_pix = max_loc[0]
    yc_pix = max_loc[1]
    
    top_right   = aia_map_94.pixel_to_world((xc_pix+npix_crop//2-1)*u.pix,(yc_pix+npix_crop//2-1)*u.pix)
    bottom_left = aia_map_94.pixel_to_world((xc_pix-npix_crop//2)*u.pix,(yc_pix-npix_crop//2)*u.pix)

    submaps.append(aia_map_94.submap(bottom_left, top_right=top_right).data)
    exptime.append(aia_map_94.meta['exptime'])
    metadata = aia_map_94.meta
    
    sorted_wavs = [131, 171, 193, 211]
    
    for aia_submap in aia_submaps[1:]:
        
        new_bl = SkyCoord(bottom_left.Tx, bottom_left.Ty, frame=aia_submap.coordinate_frame)
        new_tr = SkyCoord(top_right.Tx, top_right.Ty, frame=aia_submap.coordinate_frame)
        submaps.append(aia_submap.submap(new_bl, top_right=new_tr).data)
        exptime.append(aia_submap.meta['exptime'])
    
    return submaps, exptime, metadata, top_right, bottom_left

def normalize_exposure_and_sigma(aia_maps, exptime):
    
    # Dimensions of a single image
    dim = aia_maps[0].data.shape
    aia_img = np.zeros((len(aia_maps),dim[0],dim[1]))
    
    for i in range(len(aia_maps)):
        aia_img[i,:,:] = aia_maps[i].data
    
    sorted_wavs = [94, 131, 171, 193, 211] 
    
    # Values <= 0 are set equal to 1e-3 to avoid numerical problems
    idx = np.where(aia_img <= 0)
    if len(idx[0]) > 0:
        aia_img[idx] = 1e-3
    
    sigma = dem_rml.AIAEstimateError(aia_img, sorted_wavs)
    
    for i in range(len(sorted_wavs)):
        aia_img[i,:,:] = aia_img[i,:,:] / exptime[i]  
        sigma[i,:,:]    = sigma[i,:,:] / exptime[i]
    
    return aia_img, sigma    
    
def compute_em_map(rml_dem, logT, dT, metadata, logTmin=6.6):

    
    dim = rml_dem.shape
    idx = np.where(logT >= logTmin)
    idx = idx[0]
    pixel_area = (72528128.*0.6)**2
    
    em_map = np.zeros((dim[1], dim[2]))
    for i in range(len(idx)):
        em_map += rml_dem[idx[i],:,:]*dT[i]*pixel_area
    
    em_map = Map(em_map, metadata)
    
    return em_map


def load_realtime_XRS(goes_folder):   
    ''' Downloads real-time XRS data from NOAA, and is to be used for real-time testing and launch.
    Note: the url and filename remains the same- do not edit.
    ''' 
    json_url='https://services.swpc.noaa.gov/json/goes/primary/xrays-6-hour.json'
    json_file='xrays-6-hour.json'
    
    if os.path.exists(os.path.join(goes_folder, 'xrays-6-hour.json')):
        os.remove(os.path.join(goes_folder, 'xrays-6-hour.json'))
    wget.download(json_url, bar=None, out = goes_folder)
    with open(os.path.join(goes_folder, 'xrays-6-hour.json')) as f: 
        df = pd.DataFrame(json.load(f))
    xrsa_current = df[df.energy == '0.05-0.4nm'].iloc[-30:]
    xrsa_current.reset_index(drop=True, inplace=True)
    xrsb_current = df[df.energy == '0.1-0.8nm'] .iloc[-30:]
    xrsb_current.reset_index(drop=True, inplace=True)
    #changing time_tag to datetime format: 
    xrsa_current.loc[:,'time_tag'] = pd.to_datetime(xrsa_current.loc[:,'time_tag'], format='%Y-%m-%dT%H:%M:%SZ')#, format='ISO8601')
    xrsb_current.loc[:,'time_tag'] = pd.to_datetime(xrsb_current.loc[:,'time_tag'], format='%Y-%m-%dT%H:%M:%SZ')#, format='ISO8601')
    
    return xrsa_current, xrsb_current


def write_csv_em(file_name, time_em, total_em, function_csv='a'):
    
    header_csv = ['time_em', 'total_em']

    data = [time_em, total_em]

    with open(file_name, function_csv, encoding='UTF8', newline='') as file_csv:

            writer = csv.writer(file_csv)
            if function_csv == 'w':
                writer.writerow(header_csv)
            else:
                writer.writerow(data)


def convert_utc_to_timezone(this_datetime, timezone='US/Central'):

    utc = pytz.timezone('UTC')
    new_timezone = pytz.timezone(timezone)
    
    this_time_utc = utc.localize(this_datetime)
    this_time_new_timezone = this_time_utc.astimezone(new_timezone)
    
    return this_time_new_timezone                
                
                
def plot_results(plots_folder, aia_submaps, em_map, top_right, bottom_left, xrsa_current, xrsb_current, arnum, file_name_em_csv, 
                 current_time, timezone='US/Central'):

    xrsab_time = xrsa_current['time_tag']
    goes_time_array  = []
    
    for j in range(len(xrsab_time)):
        
            this_utc_time        = xrsab_time[j].to_pydatetime()#datetime.fromtimestamp(xrsab_time[j])
            this_new_timezone_time = convert_utc_to_timezone(this_utc_time)
            goes_time_array.append(this_new_timezone_time)
    
    goes_time_array = np.array(goes_time_array)
    goes_xrsa_flux  = xrsa_current['flux']
    goes_xrsb_flux  = xrsb_current['flux']
    
    # Plot AIA submaps
    fig, ax = plt.subplots(figsize=(22,10))

    ax1 = plt.subplot2grid((2,5), (0,0), colspan=1, projection=aia_submaps[0])
    ax2 = plt.subplot2grid((2,5), (0,1), colspan=1, projection=aia_submaps[1])
    ax3 = plt.subplot2grid((2,5), (0,2), colspan=1, projection=aia_submaps[2])
    ax4 = plt.subplot2grid((2,5), (0,3), colspan=1, projection=aia_submaps[3])
    ax5 = plt.subplot2grid((2,5), (0,4), colspan=1, projection=aia_submaps[4])
    ax6 = plt.subplot2grid((2,5), (1,0), colspan=2, projection=em_map)
    ax7 = plt.subplot2grid((2,5), (1,2), colspan=3)
    
    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
    
    # Define axes list
    ax = [ax1, ax2, ax3, ax4, ax5]
    
    labelsize = 15
    ticksize  = 15
    chsize  = 15
    legsize = 15
    xlabel = "Solar X [arcsec]"
    ylabel = "Solar Y [arcsec]"
    for jj in range(5):
        
        new_bl = SkyCoord(bottom_left.Tx, bottom_left.Ty, frame=aia_submaps[jj].coordinate_frame)
        new_tr = SkyCoord(top_right.Tx, top_right.Ty, frame=aia_submaps[jj].coordinate_frame)
        
        aia_submaps[jj].plot(axes=ax[jj])
        #aia_submaps[jj].draw_rectangle(
        aia_submaps[jj].draw_quadrangle(
            new_bl,
            axes=ax[jj],
            top_right=new_tr,
            color="red",
            linewidth=2,
        )

        ax[jj].set_title('AIA ' + str(aia_submaps[jj].meta['wavelnth']) + 'Å', fontsize=labelsize)
        ax[jj].set_xlabel(xlabel,fontsize=labelsize)
        ax[jj].set_ylabel(ylabel,fontsize=labelsize)
        ax[jj].tick_params(axis='x', labelsize=ticksize)
        ax[jj].tick_params(axis='y', labelsize=ticksize)
    
    # Plot EM map
    title  = 'AIA Emission Measure \n (T $\geq 10^{6.6}$ K)'
    em_map.plot_settings['norm'] = colors.LogNorm(vmin=1e42, vmax=1e45, clip=True)#vmax=np.max(em_map.data)
    em_map.plot_settings['cmap'] = matplotlib.colormaps['CMRmap']
    
    im = em_map.plot(axes=ax6)

    ax6.grid(False)
    ax6.set_title(title,fontsize=labelsize)
    ax6.set_xlabel(xlabel,fontsize=labelsize)
    ax6.set_ylabel(ylabel,fontsize=labelsize)
    ax6.tick_params(axis='x', labelsize=ticksize)
    ax6.tick_params(axis='y', labelsize=ticksize)

    cax = fig.add_axes([ax6.get_position().x1+0.01,ax6.get_position().y0,0.01,ax6.get_position().height])
    cbar = fig.colorbar(im,cax=cax)#,ticks=cbarticks)
    cbar.ax.tick_params(labelsize=labelsize) 
    cbar.ax.set_ylabel('EM [cm$^{-3}$ pixel$^{-1}$]',fontsize=labelsize)
    
    # Plot total EM and GOES
    em_csv   = pd.read_csv(file_name_em_csv)
    time_em  = np.array(em_csv['time_em'])
    total_em = np.array(em_csv['total_em'])

    if len(time_em) < 3:
        return 0

    # Define time array
    time_em_array = []
    for j in range(len(time_em)):
        this_ut_time   = datetime.datetime.strptime(time_em[j], '%d-%m-%YT%H:%M:%S')
        this_new_timezone_time = convert_utc_to_timezone(this_ut_time, timezone=timezone)
        time_em_array.append(this_new_timezone_time)

    time_em_array      = np.array(time_em_array)
    
    # Minimum and maximum times to be displayed in the plots
    min_time = np.max(np.array([time_em_array[-1] - timedelta(minutes=25), np.min(goes_time_array)]))
    max_time = np.max(goes_time_array)
    
    # Make plot
    ax7.plot(goes_time_array,goes_xrsa_flux, 'cyan', label='GOES XRSA')
    ax7.plot(goes_time_array,goes_xrsb_flux, 'blue', label='GOES XRSB')
    ax7.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=tz.gettz(timezone)))
    ax7.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
    ax7.set_yscale('log')
    ax7.tick_params(axis="x", labelsize=chsize)
    ax7.tick_params(axis="y", labelsize=chsize)
    ax7.set(xlabel='Time (' + time_em_array[-1].strftime("%d-%m-%Y") + ')')
    ax7.set(ylabel='GOES level')#
    ax7.set_title('AIA data time - ' + time_em_array[-1].strftime("%H:%M:%S") + ' ' + timezone, fontsize=chsize*2)
    ax7.xaxis.label.set_size(chsize)
    # ax7.set_xticks(goes_time_array[::2])
    # ax7.set_xticklabels(goes_time_array[::2], rotation=45)
    ax7.yaxis.label.set_size(chsize)
    ax7.set_xlim((min_time,max_time))
    ax7.set_ylim(1e-8, 1e-4)
    ax7.yaxis.set_ticks([1e-8, 1e-7, 1e-6, 1e-5, 1e-4], ["A", "B", "C", "M", "X"])
    ax7.grid(True)


    color = 'blue'
    ax7.tick_params(axis='y', labelcolor=color)
    ax7.yaxis.label.set_color(color)

    ax8 = ax7.twinx()
    ax8.plot(time_em_array,total_em, 'r', label='AIA EM')
    ax8.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=tz.gettz(timezone)))
    ax8.set_yscale('log')
    ax8.set_ylim(1e45, 1e49)
    ax8.set_xlim((min_time,max_time))
    ax8.tick_params(axis="x", labelsize=chsize)
    ax8.set(ylabel='EM [cm$^{-3}$]')
    ax8.tick_params(axis="y", labelsize=chsize)
    ax8.yaxis.label.set_size(chsize)
    color = 'red'
    ax8.tick_params(axis='y', labelcolor=color)
    ax8.yaxis.label.set_color(color)
    ax8.spines['right'].set_color(color)
    ax8.spines['left'].set_color('blue')
    
    fig.legend(bbox_to_anchor=(0.09, 0.05, 0.45, 0.38), fontsize=legsize)
    plt.savefig(os.path.join(plots_folder, 'aia_em_' + str(arnum)) + '_' + time_em_array[-1].strftime("%d-%m-%YT%H%M%S"), dpi=100,bbox_inches='tight')


def stream_aia_data(duration_stream, tresp, logT, dlogT, dT, data_folder, 
                    n_ar, ar_lon, ar_lat, arnum, timezone='US/Central', 
                    latency=10, reference_wav=193, logTmin=6.6):
    """
    duration_stream: int, duration of the data stream in minutes
    
    tresp: nunpy array, AIA temperature response matrix
    
    logT: numpy array containing the value of the base 10 logarithm of the center of each temparture bin
    
    dlogT: numpy array containing the value of the width of the base 10 logarithm of the temperature bins
    
    dT: numpy array containing the value of the width of the temperature bins
    
    data_folder: string, path of the folder where the data are saved
    
    n_ar: number of considered active regions
    
    ar_lon: list, contains longitude coordinates (degrees) of the active region center
    
    ar_lat: lits, contains latitude coordinates (degrees) of the active region center
    
    arnum: list, contains ID number of the considered active regions
    
    timezone: string, name of the timezone with respect to which times are expressed in the plot
    
    latency: int, minutes of data in the past that are queried every time (to be sure to get the latest data)
    
    reference_wav: int, reference wavelength to be considered for determining 12s "cycles" of AIA data
    
    logTmin: float, mimum (Log) temperature value to be considered for computing the Emission Measure map 
    """
    
    # Define JSOC server client
    client = configure_jsoc_server()
    
    # EM maps folders
    em_maps_folder = os.path.join(data_folder, 'em_maps')
    mkdir(em_maps_folder)
    # Create file with EM values over time for each AR
    for i in range(n_ar):
        file_name_em_csv = os.path.join(em_maps_folder, 'total_em_' + str(arnum[i]) + '.csv')
        if not os.path.exists(file_name_em_csv):
            write_csv_em(file_name_em_csv, 0, 0, function_csv='w')
    
    # GOES data folder
    goes_folder = os.path.join(data_folder, 'goes_data')
    mkdir(goes_folder)
    
    # Plots folder
    plots_folder = os.path.join(data_folder, 'all_plots')
    mkdir(plots_folder)
    
    # AIA data folder
    aia_data_folder = os.path.join(data_folder, 'aia_data_folder')
    mkdir(aia_data_folder)
    
    wavelengths_needed = np.array([94, 131, 171, 193, 211])
    
    start_time_ut    = datetime.datetime.now(datetime.timezone.utc) - timedelta(minutes = latency)  
    current_time_ut  = start_time_ut
    # Initialize difference between start time and current time (zero at the beginning of the stream)
    time_diff = 0
    
    # Define utc time zone
    utc = pytz.timezone('UTC')

    while time_diff <= duration_stream:

        # Make the query
        query, segments = client.query('aia.lev1_nrt2[' + current_time_ut.strftime("%Y.%m.%d_%H:%M:%S") + '_UT/' + str(latency) + 'm]',  key='T_REC, WAVELNTH', seg='image_lev1')

        # Extract wavelengths, time of the measurement, segment link 
        wavelnth = np.array(query['WAVELNTH'])
        t_rec    = np.array(query['T_REC'])
        segments = np.squeeze(np.array(segments))

        # Check if reference wavelength is present in the set of data that have been queried
        idx = np.where(wavelnth == reference_wav)
        idx = idx[0]

        if len(idx) == 0:
            print("Reference wavelength not found. Wait 15 s.")
            time.sleep(15)
            time_diff = datetime.datetime.now() - start_time_test
            time_diff = time_diff.seconds/60
            continue

        # Divide data into cycles
        grouped_wav      = []
        grouped_t_rec    = []
        grouped_segments = []
        start_time_series = []

        for start, end in zip(idx, idx[1:]):

            this_wav   = wavelnth[start:end]
            this_t_rec = t_rec[start:end]
            this_segments = segments[start:end]
            this_start_time = datetime.datetime.strptime(this_t_rec[0], '%Y-%m-%dT%H:%M:%SZ')
            start_time_series.append(utc.localize(this_start_time))

            # Remove 335 A, 304 A, 1600 A, 1700 A and 4500 A
            idx_remove = np.where((this_wav == 304) | (this_wav == 335) | (this_wav == 1600) | (this_wav == 1700) | (this_wav == 4500))
            idx_remove = idx_remove[0]
            if len(idx_remove) > 0:
                this_wav      = np.delete(this_wav, idx_remove)
                this_t_rec    = np.delete(this_t_rec, idx_remove)
                this_segments = np.delete(this_segments, idx_remove)

            grouped_wav.append(this_wav)
            grouped_t_rec.append(this_t_rec)
            grouped_segments.append(this_segments)

        
        start_time_series = np.array(start_time_series)
        
        # Take the last 12s "cycle"
        grouped_wav       = grouped_wav[-1]
        grouped_t_rec     = grouped_t_rec[-1]
        grouped_segments  = grouped_segments[-1]
        start_time_series = start_time_series[-1]
        
        # Check if last data requested is new and if all needed wavelengths are needed
        if (start_time_series >= current_time_ut) and (np.sum(np.in1d(wavelengths_needed, grouped_wav)) == len(wavelengths_needed)):
            
            t = time.time()
            aia_maps, dowloaded_data_folder, error = download_aia_data(grouped_wav, grouped_t_rec, grouped_segments, aia_data_folder, timezone=timezone)
            calibrated_aia_maps = calibrate_full_disk_maps(aia_maps)
      
            if error:
                print("Error in downloading data. Continue..")
                time.sleep(30)
                continue
            
            # Crop images around ARs and compute EM of the "hottest region"
            for i in range(n_ar):
                
                aia_submaps = crop_full_disk_maps(calibrated_aia_maps, ar_lon[i], ar_lat[i], arnum[i], dowloaded_data_folder, n_pix=1000)
                aia_submaps_dem, exptime, metadata, top_right, bottom_left = crop_around_max_aia_94A(aia_submaps, npix_crop=200)
                
                # Compute EM
                print('    Compute EM - AR: ' + str(arnum[i]))
                aia_maps_norm, sigma               = normalize_exposure_and_sigma(aia_submaps_dem, exptime)
                rml_dem, rml_dem_error, lam_pixel  = dem_rml.dem_rml(aia_maps_norm, sigma, exptime, tresp, logT, dlogT, silent=True)
        
                # Compute and save EM map
                em_map = compute_em_map(rml_dem, logT, dT, metadata, logTmin=logTmin)
                fitsname = os.path.join(em_maps_folder, 'em_map_' + start_time_series.strftime("%d-%m-%YT%H%M%S")+ \
                                        '_ar' + str(arnum[i]) + '.fits')
                sunpy.io.fits.write(fitsname, em_map.data, em_map.fits_header, overwrite=True)
                
                # Save current value of the total EM in the selected area
                file_name_em_csv = os.path.join(em_maps_folder, 'total_em_' + str(arnum[i]) + '.csv')
                write_csv_em(file_name_em_csv, start_time_series.strftime("%d-%m-%YT%H:%M:%S"), np.sum(em_map.data))
                
                # Load GOES data
                xrsa_current, xrsb_current = load_realtime_XRS(goes_folder)
                
                # Plot_results
                plot_results(plots_folder, aia_submaps, em_map, top_right, bottom_left, xrsa_current, xrsb_current, arnum[i], file_name_em_csv, 
                 start_time_series.strftime("%d-%m-%YT%H%M%S"), timezone=timezone)
                
            elapsed = time.time() - t
            print('Elapsed time: ' + str(round(elapsed)) + ' s')
            time_diff = datetime.datetime.now(datetime.timezone.utc) - start_time_ut
            time_diff = time_diff.seconds/60
            # Reset 'current_time_ut'
            current_time_ut = start_time_series
            continue
            
        else:
            print("No latest data series. Wait 15 s.")
            time.sleep(15)
            time_diff = datetime.datetime.now(datetime.timezone.utc) - start_time_ut
            time_diff = time_diff.seconds/60
            continue