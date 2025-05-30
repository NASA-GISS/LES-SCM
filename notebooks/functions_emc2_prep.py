import os, sys, tarfile
from pathlib import Path
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import glob

import time
import pylab as pl
from IPython import display
from matplotlib import colors
import pandas as pd
import emc2
import xarray as xr

from os import listdir
from os.path import isfile, join

## for questions, please contact 
## Florian Tornow: ft2544@columbia.edu

def arm_loader(inst='KAZR',DOI='2020-03-13',TOI=18.0,DELTAT=1):
    if inst=='KAZR':
        print('loading KAZR data...')        
        PATH='/ccsopen/home/floriantornow/COMBLE_ARM_DAT/anxkazrcfrgeqcM1.b1/'
    elif inst=='MPL':
        print('loading MPL data...')        
        PATH='/ccsopen/home/floriantornow/COMBLE_ARM_DAT/anxmplpolfsM1.b1/'
        
    TMIN = np.datetime64(DOI + 'T' + str(np.int(TOI)).zfill(2) + ':' + str(np.int(np.round(TOI%1*60,2))).zfill(2) + ':00') - np.int64(DELTAT*3600/2)
    TMAX = np.datetime64(DOI + 'T' + str(np.int(TOI)).zfill(2) + ':' + str(np.int(np.round(TOI%1*60,2))).zfill(2) + ':00') + np.int64(DELTAT*3600/2)
    print(TMIN)# + '<<-->>' + TMAX)
    print(TMAX)
        
    onlyfiles = [f for f in listdir(PATH) if isfile(join(PATH, f))]
    DOI_str = DOI[0:4]+DOI[5:7]+DOI[8:10]

    ds_col = []
    for f in onlyfiles:
        if not DOI_str in f:
            continue
        print(f)
        
        ## go through each file
        ds = nc.Dataset(PATH + f)        
        dataset = xr.open_dataset(xr.backends.NetCDF4DataStore(ds))
        
        ## concatenate if within limits
        for tt in range(0,len(ds['time'])):
            dat_time = np.datetime64('1970-01-01T00:00:00') + np.int64(ds['base_time'][:]) + np.int64(ds['time'][tt])
            if (dat_time > TMIN) & (dat_time < TMAX):
                if len(ds_col)==0:
                    ds_col = dataset.isel(time=tt)
                else:
                    ds_col = xr.concat([ds_col, dataset.isel(time=tt)], "time")
                #print(ds_col.time)
    return(ds_col)

def dephy_for_emc2(FILE_IN='/ccsopen/proj/atm133/dharma/emc2_folder/',test_plot=False):
    ## confirm input
    print(FILE_IN)

    ## to result in following output
    FILE_OUT = os.path.splitext(FILE_IN)[0] + "_dephy.nc"
    print(FILE_OUT)
    
    
    ## take 3D DEPHY-style output and transform into EMC^2-friendly dataset   
    ## ...
    ds = nc.Dataset(FILE_IN)
    print(ds)
    z = ds.variables['height'][:].data
    t = ds.variables['time'][:].data
    
    z_num = len(z)
    x = ds.variables['x'][:].data
    y = ds.variables['y'][:].data

    ## clear if file already exists
    if(os.path.isfile(FILE_OUT)):
        os.remove(FILE_OUT)
        
    ncfile = nc.Dataset(FILE_OUT,mode='w',format='NETCDF4_CLASSIC')
    print(ncfile)
    x_dim = ncfile.createDimension('x',len(x))
    y_dim = ncfile.createDimension('y',len(y))
    hgt_dim = ncfile.createDimension('hgt',z_num)
    
    x_var = ncfile.createVariable('x', np.float64, ('x',))
    x_var.units = ''
    x_var.long_name = 'x dimension - can be used in the simulator as the time domain'
    x_var.Missing_value = '-9999'
    x_var[:] = x[:]
    
    y_var = ncfile.createVariable('y', np.float64, ('y',))
    y_var.units = ''
    y_var.long_name = 'y dimension'
    y_var.Missing_value = '-9999'
    y_var[:] = y[:]
    
    hgt_var = ncfile.createVariable('hgt', np.float32, ('hgt',))
    hgt_var.units = 'm'
    hgt_var.long_name = 'Height'
    hgt_var.Missing_value = '-9999'
    hgt_var[:] = z[:]
    
    z_var = ncfile.createVariable('z', np.float32, ('hgt','y','x'))
    z_var.units = 'm'
    z_var.long_name = 'Altitude'
    z_var.Missing_value = '-9999'
    z_var[:] = np.tile(z, (len(x), len(y), 1)).transpose()[:]
    
    u_var = ncfile.createVariable('u_wind', np.float32, ('hgt','y','x'))
    u_var.units = 'm/s'
    u_var.long_name = 'u wind'
    u_var.Missing_value = '-9999'
    u_var[:] = ds.variables['ua'][:].data
       
    v_var = ncfile.createVariable('v_wind', np.float32, ('hgt','y','x'))
    v_var.units = 'm/s'
    v_var.long_name = 'v wind'
    v_var.Missing_value = '-9999'
    v_var[:] = ds.variables['va'][:].data
    
    w_var = ncfile.createVariable('w_wind', np.float32, ('hgt','y','x'))
    w_var.units = 'm/s'
    w_var.long_name = 'w wind'
    w_var.Missing_value = '-9999'
    w_var[:] = ds.variables['wa'][:].data
    
    qc_var = ncfile.createVariable('qcl', np.float32, ('hgt','y','x'))
    qc_var.units = 'kg/kg'
    qc_var.long_name = 'Cloud liquid mixing ratio'
    qc_var.Missing_value = '-9999'
    qc_var[:] = ds.variables['qlc'][:].data
    
    qic_var = ncfile.createVariable('qci', np.float32, ('hgt','y','x'))
    qic_var.units = 'kg/kg'
    qic_var.long_name = 'Cloud ice mixing ratio'
    qic_var.Missing_value = '-9999'
    qic_var[:] = ds.variables['qic'][:].data
    
    qr_var = ncfile.createVariable('qpl', np.float32, ('hgt','y','x'))
    qr_var.units = 'kg/kg'
    qr_var.long_name = 'Rain mixing ratio'
    qr_var.Missing_value = '-9999'
    qr_var[:] = ds.variables['qlr'][:].data
    
    qif_var = ncfile.createVariable('qpi', np.float32, ('hgt','y','x'))
    qif_var.units = 'kg/kg'
    qif_var.long_name = 'Snow mixing ratio'
    qif_var.Missing_value = '-9999'
    qif_var[:] = ds.variables['qis'][:].data
    
    qid_var = ncfile.createVariable('qpir', np.float32, ('hgt','y','x'))
    qid_var.units = 'kg/kg'
    qid_var.long_name = 'Rimed ice mixing ratio'
    qid_var.Missing_value = '-9999'
    qid_var[:] = ds.variables['qig'][:].data
    
    nc_var = ncfile.createVariable('ncl', np.float32, ('hgt','y','x'))
    nc_var.units = 'cm^-3'
    nc_var.long_name = 'Cloud liquid number concentration'
    nc_var.Missing_value = '-9999'
    nc_var[:] = ds.variables['nlc'][:].data/1e6
    
    nic_var = ncfile.createVariable('nci', np.float32, ('hgt','y','x'))
    nic_var.units = 'cm^-3'
    nic_var.long_name = 'Cloud ice number concentration'
    nic_var.Missing_value = '-9999'
    nic_var[:] = ds.variables['nic'][:].data/1e6
    
    nr_var = ncfile.createVariable('npl', np.float32, ('hgt','y','x'))
    nr_var.units = 'cm^-3'
    nr_var.long_name = 'Rain number concentration'
    nr_var.Missing_value = '-9999'
    nr_var[:] = ds.variables['nlr'][:].data/1e6
    
    nif_var = ncfile.createVariable('npi', np.float32, ('hgt','y','x'))
    nif_var.units = 'cm^-3'
    nif_var.long_name = 'Snow number concentration'
    nif_var.Missing_value = '-9999'
    nif_var[:] = ds.variables['nis'][:].data/1e6
    
    nid_var = ncfile.createVariable('npir', np.float32, ('hgt','y','x'))
    nid_var.units = 'cm^-3'
    nid_var.long_name = 'Rimed ice number concentration'
    nid_var.Missing_value = '-9999'
    nid_var[:] = ds.variables['nig'][:].data/1e6
    
    qv_var = ncfile.createVariable('q', np.float32, ('hgt','y','x'))
    qv_var.units = 'kg/kg'
    qv_var.long_name = 'Specific humidity'
    qv_var.Missing_value = '-9999'
    qv_var[:] = ds.variables['qv'][:].data
    
    p_var = ncfile.createVariable('p', np.float32, ('hgt','y','x'))
    p_var.units = 'hPa'
    p_var.long_name = 'Pressure'
    p_var.Missing_value = '-9999'
    p_col = ds.variables['pa'][:].data[0]/100
    p_3d  = np.tile(p_col.reshape(len(z),1,1),[len(x),len(y)])
    p_var[:] = p_3d
    
    t_var = ncfile.createVariable('t', np.float32, ('hgt','y','x'))
    t_var.units = 'K'
    t_var.long_name = 'Temperature'
    t_var.Missing_value = '-9999'
    t_var[:] = ds.variables['ta'][:].data
    
    snc_var = ncfile.createVariable('strat_cl_frac', np.float32, ('hgt','y','x'))
    snc_var.units = ''
    snc_var.long_name = 'Stratiform cl fraction'
    snc_var.Missing_value = '-9999'
    snc_var[:] = ds.variables['nlc'][:].data>0
    
    snic_var = ncfile.createVariable('strat_ci_frac', np.float32, ('hgt','y','x'))
    snic_var.units = ''
    snic_var.long_name = 'Stratiform ci fraction'
    snic_var.Missing_value = '-9999'
    snic_var[:] = ds.variables['nic'][:].data>0
    
    snr_var = ncfile.createVariable('strat_pl_frac', np.float32, ('hgt','y','x'))
    snr_var.units = ''
    snr_var.long_name = 'Stratiform pl fraction'
    snr_var.Missing_value = '-9999'
    snr_var[:] = ds.variables['nlr'][:].data>0
    
    snif_var = ncfile.createVariable('strat_pi_frac', np.float32, ('hgt','y','x'))
    snif_var.units = ''
    snif_var.long_name = 'Stratiform pi fraction'
    snif_var.Missing_value = '-9999'
    snif_var[:] = ds.variables['nis'][:].data>0
    
    snid_var = ncfile.createVariable('strat_pir_frac', np.float32, ('hgt','y','x'))
    snid_var.units = ''
    snid_var.long_name = 'Stratiform pir (rimed ice) fraction'
    snid_var.Missing_value = '-9999'
    snid_var[:] = ds.variables['nig'][:].data>0
    
    conv_var = ncfile.createVariable('conv_dat', np.float32, ('hgt','y','x'))
    conv_var.units = ''
    conv_var.long_name = 'Convective data (an array of zeros to allow running in EMC2)'
    conv_var.Missing_value = '-9999'
    conv_var[:] = ds.variables['nis'][:].data*0.0
    
    ncfile.close(); print('Dataset is closed!')
    ds.close()
    
    ## output quick test
    #print(xr.open_dataset(FILE_OUT))

    
    
def dharma_for_emc2(PATH='/ccsopen/proj/atm133/dharma/emc2_folder/',SUBPATH='0-20h/',PLT_times=['025285'],test_plot=False):
    print(PATH)
    ## take DHARMA LES output and transform into EMC^2-friendly dataset   
    ## ...
    
    PATH_PLT = PATH + 'plt_files/'
    
    Rd = 287;
    Cp = 1005.7;
    Rd_2_Cp = Rd / Cp;
    soundings_filename = 'dharma.soundings.cdf'
    
    th_3d = threed_loader(INDIR=PATH_PLT,OUTDIR=PATH_PLT+'tmp/',PLT_FILE = 'dharma_PLT_'+PLT_times[0]+'.tgz',vbase= 'th')
    press_3d = threed_loader(INDIR=PATH_PLT,OUTDIR=PATH_PLT+'tmp/',PLT_FILE = 'dharma_PLT_'+PLT_times[0]+'.tgz',vbase= 'press')
    
    ds = nc.Dataset(PATH + SUBPATH + soundings_filename)
    z = ds.variables['zt'][:].data
    t = ds.variables['time'][:].data
    theta_o = ds.theta_00
    T_snd = ds.variables['T'][:].data
    th_snd = theta_o * (1 + ds.variables['th'][:].data)
    rho_a = ds.variables['rhobar'][:].data
    pi_snd = T_snd[1,:]/th_snd[1,:]
    penv_mks = 1e6 * pi_snd ** (1/ Rd_2_Cp) * 0.1 #Pa
    
    T_3d = np.tile(pi_snd, (ds.nx, ds.ny, 1)).transpose() * theta_o * (1. + th_3d);
    P_3d = np.tile(penv_mks, (ds.nx, ds.ny, 1)).transpose() + np.tile(rho_a[1,:],(ds.nx, ds.ny, 1)).transpose() * press_3d;

    if(test_plot):        
        ## set up image plot
        fig, ax = plt.subplots() 
        shw = ax.imshow(T_3d[0,:,:])
        # make bar 
        bar = plt.colorbar(shw) 
        # show plot with labels 
        plt.xlabel('X Label') 
        plt.ylabel('Y Label') 
        bar.set_label('ColorBar') 
        plt.show() 
    
    u_3d  = threed_loader(INDIR=PATH_PLT,OUTDIR=PATH_PLT+'tmp/',PLT_FILE = 'dharma_PLT_'+PLT_times[0]+'.tgz',vbase = 'u')
    v_3d  = threed_loader(INDIR=PATH_PLT,OUTDIR=PATH_PLT+'tmp/',PLT_FILE = 'dharma_PLT_'+PLT_times[0]+'.tgz',vbase = 'v')
    w_3d  = threed_loader(INDIR=PATH_PLT,OUTDIR=PATH_PLT+'tmp/',PLT_FILE = 'dharma_PLT_'+PLT_times[0]+'.tgz',vbase = 'w')
    
    qc_3d = threed_loader(INDIR=PATH_PLT,OUTDIR=PATH_PLT+'tmp/',PLT_FILE = 'dharma_PLT_'+PLT_times[0]+'.tgz',vbase = 'qc')
    qv_3d = threed_loader(INDIR=PATH_PLT,OUTDIR=PATH_PLT+'tmp/',PLT_FILE = 'dharma_PLT_'+PLT_times[0]+'.tgz',vbase = 'qv')
    qr_3d = threed_loader(INDIR=PATH_PLT,OUTDIR=PATH_PLT+'tmp/',PLT_FILE = 'dharma_PLT_'+PLT_times[0]+'.tgz',vbase = 'qr')
    qic_3d = threed_loader(INDIR=PATH_PLT,OUTDIR=PATH_PLT+'tmp/',PLT_FILE = 'dharma_PLT_'+PLT_times[0]+'.tgz',vbase = 'qic')
    qif_3d = threed_loader(INDIR=PATH_PLT,OUTDIR=PATH_PLT+'tmp/',PLT_FILE = 'dharma_PLT_'+PLT_times[0]+'.tgz',vbase = 'qif')
    qid_3d = threed_loader(INDIR=PATH_PLT,OUTDIR=PATH_PLT+'tmp/',PLT_FILE = 'dharma_PLT_'+PLT_times[0]+'.tgz',vbase = 'qid')
    
    nc_3d = threed_loader(INDIR=PATH_PLT,OUTDIR=PATH_PLT+'tmp/',PLT_FILE = 'dharma_PLT_'+PLT_times[0]+'.tgz',vbase = 'nc')
    nr_3d = threed_loader(INDIR=PATH_PLT,OUTDIR=PATH_PLT+'tmp/',PLT_FILE = 'dharma_PLT_'+PLT_times[0]+'.tgz',vbase = 'nr')
    nic_3d = threed_loader(INDIR=PATH_PLT,OUTDIR=PATH_PLT+'tmp/',PLT_FILE = 'dharma_PLT_'+PLT_times[0]+'.tgz',vbase = 'nic')
    nif_3d = threed_loader(INDIR=PATH_PLT,OUTDIR=PATH_PLT+'tmp/',PLT_FILE = 'dharma_PLT_'+PLT_times[0]+'.tgz',vbase = 'nif')
    nid_3d = threed_loader(INDIR=PATH_PLT,OUTDIR=PATH_PLT+'tmp/',PLT_FILE = 'dharma_PLT_'+PLT_times[0]+'.tgz',vbase = 'nid')
    
    z_num = len(z)
    x = np.linspace(-( ds.L_x / 2 - np.double(ds.L_x / ds.nx) / 2), ds.L_x / 2 - np.double(ds.L_x / ds.nx) / 2 , ds.nx);
    y = np.linspace(-( ds.L_y / 2 - np.double(ds.L_y / ds.ny) / 2), ds.L_y / 2 - np.double(ds.L_y / ds.ny) / 2 , ds.ny);

    ## clear if file already exists
    if(os.path.isfile(PATH + 'emc2_prep_'+PLT_times[0]+'.nc')):
        os.remove(PATH + 'emc2_prep_'+PLT_times[0]+'.nc')
        
    ncfile = nc.Dataset(PATH + 'emc2_prep_'+PLT_times[0]+'.nc',mode='w',format='NETCDF4_CLASSIC')
    print(ncfile)
    x_dim = ncfile.createDimension('x',len(x))
    y_dim = ncfile.createDimension('y',len(y))
    hgt_dim = ncfile.createDimension('hgt',z_num)
    
    x_var = ncfile.createVariable('x', np.float64, ('x',))
    x_var.units = ''
    x_var.long_name = 'x dimension - can be used in the simulator as the time domain'
    x_var.Missing_value = '-9999'
    x_var[:] = x[:]
    
    y_var = ncfile.createVariable('y', np.float64, ('y',))
    y_var.units = ''
    y_var.long_name = 'y dimension'
    y_var.Missing_value = '-9999'
    y_var[:] = y[:]
    
    hgt_var = ncfile.createVariable('hgt', np.float32, ('hgt',))
    hgt_var.units = 'm'
    hgt_var.long_name = 'Height'
    hgt_var.Missing_value = '-9999'
    hgt_var[:] = z[:]
    
    z_var = ncfile.createVariable('z', np.float32, ('hgt','y','x'))
    z_var.units = 'm'
    z_var.long_name = 'Altitude'
    z_var.Missing_value = '-9999'
    z_var[:] = np.tile(z, (ds.nx, ds.ny, 1)).transpose()[:]
    
    u_var = ncfile.createVariable('u_wind', np.float32, ('hgt','y','x'))
    u_var.units = 'm/s'
    u_var.long_name = 'u wind'
    u_var.Missing_value = '-9999'
    u_var[:] = u_3d[:]
    
    v_var = ncfile.createVariable('v_wind', np.float32, ('hgt','y','x'))
    v_var.units = 'm/s'
    v_var.long_name = 'v wind'
    v_var.Missing_value = '-9999'
    v_var[:] = v_3d[:]
    
    w_var = ncfile.createVariable('w_wind', np.float32, ('hgt','y','x'))
    w_var.units = 'm/s'
    w_var.long_name = 'w wind'
    w_var.Missing_value = '-9999'
    w_var[:] = w_3d[:]
    
    qc_var = ncfile.createVariable('qcl', np.float32, ('hgt','y','x'))
    qc_var.units = 'kg/kg'
    qc_var.long_name = 'Cloud liquid mixing ratio'
    qc_var.Missing_value = '-9999'
    qc_var[:] = qc_3d[:]
    
    qic_var = ncfile.createVariable('qci', np.float32, ('hgt','y','x'))
    qic_var.units = 'kg/kg'
    qic_var.long_name = 'Cloud ice mixing ratio'
    qic_var.Missing_value = '-9999'
    qic_var[:] = qic_3d[:]
    
    qr_var = ncfile.createVariable('qpl', np.float32, ('hgt','y','x'))
    qr_var.units = 'kg/kg'
    qr_var.long_name = 'Rain mixing ratio'
    qr_var.Missing_value = '-9999'
    qr_var[:] = qr_3d[:]
    
    qif_var = ncfile.createVariable('qpi', np.float32, ('hgt','y','x'))
    qif_var.units = 'kg/kg'
    qif_var.long_name = 'Snow mixing ratio'
    qif_var.Missing_value = '-9999'
    qif_var[:] = qif_3d[:]
    
    qid_var = ncfile.createVariable('qpir', np.float32, ('hgt','y','x'))
    qid_var.units = 'kg/kg'
    qid_var.long_name = 'Rimed ice mixing ratio'
    qid_var.Missing_value = '-9999'
    qid_var[:] = qid_3d[:]
    
    nc_var = ncfile.createVariable('ncl', np.float32, ('hgt','y','x'))
    nc_var.units = 'cm^-3'
    nc_var.long_name = 'Cloud liquid number concentration'
    nc_var.Missing_value = '-9999'
    nc_var[:] = nc_3d[:]/1e6
    
    nic_var = ncfile.createVariable('nci', np.float32, ('hgt','y','x'))
    nic_var.units = 'cm^-3'
    nic_var.long_name = 'Cloud ice number concentration'
    nic_var.Missing_value = '-9999'
    nic_var[:] = nic_3d[:]/1e6
    
    nr_var = ncfile.createVariable('npl', np.float32, ('hgt','y','x'))
    nr_var.units = 'cm^-3'
    nr_var.long_name = 'Rain number concentration'
    nr_var.Missing_value = '-9999'
    nr_var[:] = nr_3d[:]/1e6
    
    nif_var = ncfile.createVariable('npi', np.float32, ('hgt','y','x'))
    nif_var.units = 'cm^-3'
    nif_var.long_name = 'Snow number concentration'
    nif_var.Missing_value = '-9999'
    nif_var[:] = nif_3d[:]/1e6
    
    nid_var = ncfile.createVariable('npir', np.float32, ('hgt','y','x'))
    nid_var.units = 'cm^-3'
    nid_var.long_name = 'Rimed ice number concentration'
    nid_var.Missing_value = '-9999'
    nid_var[:] = nid_3d[:]/1e6
    
    qv_var = ncfile.createVariable('q', np.float32, ('hgt','y','x'))
    qv_var.units = 'kg/kg'
    qv_var.long_name = 'Specific humidity'
    qv_var.Missing_value = '-9999'
    qv_var[:] = qv_3d[:]/1e6
    
    p_var = ncfile.createVariable('p', np.float32, ('hgt','y','x'))
    p_var.units = 'hPa'
    p_var.long_name = 'Pressure'
    p_var.Missing_value = '-9999'
    p_var[:] = P_3d[:]/100
    
    t_var = ncfile.createVariable('t', np.float32, ('hgt','y','x'))
    t_var.units = 'K'
    t_var.long_name = 'Temperature'
    t_var.Missing_value = '-9999'
    t_var[:] = T_3d[:]
    
    snc_var = ncfile.createVariable('strat_cl_frac', np.float32, ('hgt','y','x'))
    snc_var.units = ''
    snc_var.long_name = 'Stratiform cl fraction'
    snc_var.Missing_value = '-9999'
    snc_var[:] = nc_3d[:]>0
    
    snic_var = ncfile.createVariable('strat_ci_frac', np.float32, ('hgt','y','x'))
    snic_var.units = ''
    snic_var.long_name = 'Stratiform ci fraction'
    snic_var.Missing_value = '-9999'
    snic_var[:] = nic_3d[:]>0
    
    snr_var = ncfile.createVariable('strat_pl_frac', np.float32, ('hgt','y','x'))
    snr_var.units = ''
    snr_var.long_name = 'Stratiform pl fraction'
    snr_var.Missing_value = '-9999'
    snr_var[:] = nr_3d[:]>0
    
    snif_var = ncfile.createVariable('strat_pi_frac', np.float32, ('hgt','y','x'))
    snif_var.units = ''
    snif_var.long_name = 'Stratiform pi fraction'
    snif_var.Missing_value = '-9999'
    snif_var[:] = nif_3d[:]>0
    
    snid_var = ncfile.createVariable('strat_pir_frac', np.float32, ('hgt','y','x'))
    snid_var.units = ''
    snid_var.long_name = 'Stratiform pir (rimed ice) fraction'
    snid_var.Missing_value = '-9999'
    snid_var[:] = nid_3d[:]>0
    
    conv_var = ncfile.createVariable('conv_dat', np.float32, ('hgt','y','x'))
    conv_var.units = ''
    conv_var.long_name = 'Convective data (an array of zeros to allow running in EMC2)'
    conv_var.Missing_value = '-9999'
    conv_var[:] = nid_3d[:]*0.0
    
    ncfile.close(); print('Dataset is closed!')
    ds.close()
    
    

def threed_loader(PLT_FILE,INDIR='/ccsopen/proj/atm133/dharma/emc2_folder/PLT/',OUTDIR='/ccsopen/proj/atm133/dharma/emc2_folder/PLT/tmp/',vbase='w'):
    #print('3D loader: ' + INDIR)
    print('3D loading...' + vbase)
    tar = tarfile.open(INDIR + PLT_FILE)
    tar.extractall(OUTDIR)
    tar.close()
    
    ## gather info from first file
    PLT_FILE_TMP = OUTDIR + Path(PLT_FILE).stem + '_0000.cdf'
    ds = nc.Dataset(PLT_FILE_TMP)
    #for dim in ds.dimensions.items():
    #    print(dim)
    #print(ds)
    pmap = ds.pmap
    nbox = ds.nboxes
    nx = ds.nx
    ny = ds.ny
    nz = ds.nz
    ds.close()
    var_big = np.zeros((nz,ny,nx),dtype=float)
    
    ## read and organize
    for j in range(len(pmap)):
        vname = vbase + '_' + str("%04d" % (j+1,))
        
        ib = pmap[j]
        PLT_FILE_TMP = OUTDIR + Path(PLT_FILE).stem + '_' + str("%04d" % (ib,)) + '.cdf'
        #print(PLT_FILE_TMP)
        ds = nc.Dataset(PLT_FILE_TMP)
        varF  = ds.variables[vname]
        #print(varF.dimensions)
        bnds = varF.bnds
        var = varF[:]
        ngrow = varF.ngrow
        ishift = varF.ishift
        jshift = varF.jshift
        kshift = varF.kshift
        
        ib = bnds[0]
        jb = bnds[1]
        kb = bnds[2]
        ie = bnds[3]
        je = bnds[4]
        ke = bnds[5]
        mx = ie - ib + 1
        my = je - jb + 1
        mz = ke - kb + 1
        var = np.reshape(var,(mz,my,mx)) 
        var_big[(kb-1+ngrow):(ke-ngrow-kshift),(jb-1+ngrow):(je-ngrow-jshift),(ib-1+ngrow):(ie-ngrow-ishift)] = var[ngrow:(mz-ngrow-kshift),ngrow:(my-ngrow-jshift),ngrow:(mx-ngrow-ishift)]
        ds.close()
    
    ## tidy up
    files = glob.glob(OUTDIR + '/*cdf')
    for f in files:
        os.remove(f)
    
    return var_big

