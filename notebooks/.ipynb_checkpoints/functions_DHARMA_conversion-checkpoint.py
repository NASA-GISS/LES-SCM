import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import netCDF4
import datetime as dt
from netCDF4 import Dataset
import sys

## Specify directory locations


def DHARMA_convert(path,output_filename,verbose=False):

    ## translated from COMBLE notebook written by Ann Fridlind (https://github.com/ARM-Development/comble-mip/blob/main/notebooks/conversion_output/convert_DHARMA_LES_output_to_dephy_format.ipynb)

    print('--------------------')
    
    # specify start time of simulation and simulation name
    ######################################################
    if '20200313' in output_filename:
        start_dtime = '2020-03-12 22:00:00.0'
    if '20200409' in output_filename:
        start_dtime = '2020-04-08 11:00:00.0'
    if '20200425' in output_filename:
        start_dtime = '2020-04-24 15:00:00.0'
    if '20200512' in output_filename:
        start_dtime = '2020-05-11 08:00:00.0'
    
    # FixN with no ice test
    my_readdir = path
    my_outfile = output_filename

    my_rundir = '/home/tflorian/nobackup/dharma_run/cao_2018/' + my_readdir + '/'

    my_outdirs = sorted([f for f in os.listdir(my_rundir) if f.endswith('h')], key=str.lower)
    if verbose:
        print(my_outdirs)

    # specify Github scratch directory where processed model output will be committed
    my_gitdir = '/home/tflorian/LES-SCM/output_les/dharma/'
    
    # read in set-up parameters
    ####################################################### 
    #read in DHARMA parameter settings
    input_filename = my_rundir + my_outdirs[0] + '/dharma.cdf'
    dharma_params = xr.open_dataset(input_filename)
    if verbose:
        print(input_filename)

    # check if the run contains ice variables
    do_ice = bool(dharma_params['Cond'].do_ice)
    if verbose:
        print('do_ice = ',do_ice)
    
    # check for prognostic aerosol
    do_progNa = bool(dharma_params['Cond'].do_prog_na)
    if verbose:
        print('do_progNa = ',do_progNa)

    # full parameter list
    if verbose:
        dharma_params

    # read in set-up parameters
    ####################################################### 
    
    # concatenate DHARMA domain-mean instantaneous profiles and take 10-min average:
    # resample-average before concatenating and removing duplicates
    for index, elem in enumerate(my_outdirs):
        input_filename = my_rundir + elem + '/dharma.soundings.cdf'
        if verbose:
            print(input_filename)
        if index==0:
            dharma_snds = xr.open_dataset(input_filename)
            dharma_snds['time'] = pd.to_datetime(dharma_snds['time'].values, unit='s', origin=pd.Timestamp(start_dtime))
            dharma_snds = dharma_snds.resample(time="600s",closed="right",label="right").mean()
        else:
            dharma_snds2 = xr.open_dataset(input_filename)
            dharma_snds2['time'] = pd.to_datetime(dharma_snds2['time'].values, unit='s', origin=pd.Timestamp(start_dtime))
            dharma_snds2 = dharma_snds2.resample(time="600s",closed="right",label="right").mean()
            dharma_snds = xr.concat([dharma_snds,dharma_snds2],dim='time')
    dharma_snds = dharma_snds.drop_duplicates('time',keep='first')
    
    if verbose:
        dharma_snds

    # profile unit conversions and sundry
    ####################################################### 
    
    # create a dummy sounding and initialize some variables needed
    dummy_snd = dharma_snds['qc']*0.
    nz = dharma_params['geometry'].nz
    dz = dharma_snds['zw'].data[1:nz+1]-dharma_snds['zw'].data[0:nz]
    cp = 1004.
    Lhe = 2.50e6
    Lhs = Lhe + 3.34e5
    
    # compute some intermediate quantities for use below
    Fql_turb = dharma_snds['Fqc_turb'].data+dharma_snds['Fqr_turb'].data
    if do_ice:
        Fqi_turb = dharma_snds['Fqic_turb'].data+dharma_snds['Fqif_turb'].data+dharma_snds['Fqid_turb'].data
    wql_tot = 0.5*(Fql_turb[:,0:nz]+Fql_turb[:,1:nz+1])/dharma_snds['rhobar'].data+dharma_snds['WQL'].data

    if do_ice:
        wqi_tot = 0.5*(Fqi_turb[:,0:nz]+Fqi_turb[:,1:nz+1])/dharma_snds['rhobar'].data+dharma_snds['WQI'].data
    else:
        wqi_tot = np.nan*wql_tot
    PFql = dharma_snds['PFqc'].data+dharma_snds['PFqr'].data
    if do_ice:
        PFqi = dharma_snds['PFqic'].data+dharma_snds['PFqif'].data+dharma_snds['PFqid'].data
    else:
        PFqi = np.nan*PFql
    wpfl = 0.5*(PFql[:,0:nz]+PFql[:,1:nz+1])*-1./3600./dharma_snds['rhobar'].data
    wpfi = 0.5*(PFqi[:,0:nz]+PFqi[:,1:nz+1])*-1./3600./dharma_snds['rhobar'].data
    PFqc = dharma_snds['PFqc'].data 
    PFqr = dharma_snds['PFqr'].data 
    if do_ice:
        PFqic = dharma_snds['PFqic'].data 
        PFqif = dharma_snds['PFqif'].data 
        PFqid = dharma_snds['PFqid'].data 
    if do_progNa:
        ssa_sfc = (dharma_snds['Sna_1_sfc'].data[:,0]+dharma_snds['Sna_2_sfc'].data[:,0]+dharma_snds['Sna_3_sfc'].data[:,0])*dharma_snds['zw'].data[1]
    Flwd = dharma_snds['Flw_dn'].data
    Flwu = dharma_snds['Flw_up'].data
    Fnlw = Flwu - Flwd
    Suvar_adv = dharma_snds['Su2_adv'].data - dharma_snds['Subar2_adv'].data
    Svvar_adv = dharma_snds['Sv2_adv'].data - dharma_snds['Svbar2_adv'].data
    Swvar_adv = dharma_snds['Sw2_adv'].data - dharma_snds['Swbar2_adv'].data
    Stke_a = Suvar_adv + Svvar_adv + 0.5*(Swvar_adv[:,0:nz]+Swvar_adv[:,1:nz+1])
    Stke_adv_dis = dharma_snds['Stke_adv'].data + dharma_snds['Sprod'].data
    Smke = (dharma_snds['u'].data-dharma_params.translate.u)*dharma_snds['Suavg_SGS'].data + (dharma_snds['v'].data-dharma_params.translate.v)*dharma_snds['Svavg_SGS'].data
    Ske = dharma_snds['Su2avg_SGS'].data+dharma_snds['Sv2avg_SGS'].data+dharma_snds['Sw2avg_SGS'].data
    Stke_dis = Smke - Ske

    # append new variables to the data structure
    dharma_snds = dharma_snds.assign(theta = dummy_snd + (dharma_snds['th'].data+1)*dharma_snds.theta_00)
    dharma_snds = dharma_snds.assign(pi = dummy_snd + dharma_snds['T'].data/dharma_snds['theta'].data)
    dharma_snds = dharma_snds.assign(pressure = dummy_snd + np.power(dharma_snds['pi'].data,7./2)*np.power(10.,5))
    dharma_snds = dharma_snds.assign(PF = dummy_snd + 0.5*(PFqc[:,0:nz]+PFqc[:,1:nz+1]) + 0.5*(PFqr[:,0:nz]+PFqr[:,1:nz+1]))
    dharma_snds = dharma_snds.assign(nlcic = dummy_snd + dharma_snds['nc_cld'].data*1.e6/dharma_snds['rhobar'].data)
    if do_ice:
        dharma_snds = dharma_snds.assign(niic = dummy_snd + dharma_snds['ni_cld'].data*1.e6/dharma_snds['rhobar'].data)
    if do_ice:
        dharma_snds = dharma_snds.assign(PFi = dummy_snd + 0.5*(PFqic[:,0:nz]+PFqic[:,1:nz+1]) + 0.5*(PFqif[:,0:nz]+PFqif[:,1:nz+1]) + 0.5*(PFqid[:,0:nz]+PFqid[:,1:nz+1]) )
        dharma_snds['PF'] += dharma_snds['PFi']
    else:
        dharma_snds['RHI'] = np.nan*dharma_snds['RH']
        dharma_snds['PFi'] = np.nan*dharma_snds['PF']
    dharma_snds = dharma_snds.assign(uw_zt = dummy_snd + 0.5*(dharma_snds['txz_tot'].data[:,0:nz]+dharma_snds['txz_tot'].data[:,1:nz+1]))
    dharma_snds = dharma_snds.assign(vw_zt = dummy_snd + 0.5*(dharma_snds['tyz_tot'].data[:,0:nz]+dharma_snds['tyz_tot'].data[:,1:nz+1]))
    dharma_snds = dharma_snds.assign(w2_zt = dummy_snd + (dharma_snds['w2'].data[:,0:nz]+dharma_snds['w2'].data[:,1:nz+1])) # w2 = 0.5*w'2
    dharma_snds = dharma_snds.assign(wth_zt = dummy_snd + 0.5*(dharma_snds['qhz_tot'].data[:,0:nz] + dharma_snds['qhz_tot'].data[:,1:nz+1])*dharma_snds.theta_00)
    dharma_snds = dharma_snds.assign(wthli_zt = dummy_snd + dharma_snds['wth_zt'].data
                    - wql_tot*Lhe/(dharma_snds['pi'].data*cp) - wqi_tot*Lhs/(dharma_snds['pi'].data*cp)
                    - wpfl*Lhe/(dharma_snds['pi'].data*cp) - wpfi*Lhs/(dharma_snds['pi'].data*cp))                                 
    dharma_snds = dharma_snds.assign(wqv_zt = dummy_snd + 0.5*(dharma_snds['qqz_tot'].data[:,0:nz]+dharma_snds['qqz_tot'].data[:,1:nz+1]))
    dharma_snds = dharma_snds.assign(wqt_zt = dummy_snd + dharma_snds['wqv_zt'].data + wql_tot + wqi_tot + wpfl + wpfi)
    dharma_snds = dharma_snds.assign(eps = dummy_snd + Stke_dis + Stke_adv_dis)
    dharma_snds = dharma_snds.assign(LWdn = dummy_snd + 0.5*(Flwd[:,0:nz]+Flwd[:,1:nz+1]))
    dharma_snds = dharma_snds.assign(LWup = dummy_snd + 0.5*(Flwu[:,0:nz]+Flwu[:,1:nz+1]))
    dharma_snds = dharma_snds.assign(HRlw = dummy_snd + 0.5*(Fnlw[:,0:nz]+Fnlw[:,1:nz+1])/dz/dharma_snds['rhobar'].data)
    dharma_snds = dharma_snds.assign(dth_micro = dummy_snd + dharma_snds['Sth_micro'].data + dharma_snds['Sth_cond'].data)
    dharma_snds = dharma_snds.assign(dq_micro = dummy_snd + dharma_snds['Sqv_micro'].data + dharma_snds['Sqv_cond'].data)
    #dharma_snds = dharma_snds.assign(dth_turb = dummy_snd + (dharma_snds['qhz_tot'].data[:,0:nz] - 
    #                                       dharma_snds['qhz_tot'].data[:,1:nz+1])*dharma_snds.theta_00)
    #dharma_snds = dharma_snds.assign(dq_turb = dummy_snd + dharma_snds['qqz_tot'].data[:,0:nz] - dharma_snds['qqz_tot'].data[:,1:nz+1])
    dharma_snds = dharma_snds.assign(dth_turb = dummy_snd + (dharma_snds['qhz_tot'].data[:,0:nz] - dharma_snds['qhz_tot'].data[:,1:nz+1])/dz)
    dharma_snds = dharma_snds.assign(dq_turb = dummy_snd + (dharma_snds['qqz_tot'].data[:,0:nz] - dharma_snds['qqz_tot'].data[:,1:nz+1])/dz)

    if do_progNa:
        dharma_snds = dharma_snds.assign(na_loss_liq = dummy_snd + dharma_snds['na_loss_prof'].data - dharma_snds['na_loss_ice'].data)
        dharma_snds = dharma_snds.assign(dna_mixing = dummy_snd + dharma_snds['Sna_1_adv'].data + dharma_snds['Sna_2_adv'].data + dharma_snds['Sna_3_adv'].data + 
                                dharma_snds['Sna_1_sfc'].data + dharma_snds['Sna_2_sfc'].data + dharma_snds['Sna_3_sfc'].data)

    if verbose:
        dharma_snds

    # read domain-mean scalars
    ####################################################### 

    for index, elem in enumerate(my_outdirs):
        input_filename = my_rundir + elem + '/dharma.scalars.cdf'
        if verbose:
            print(input_filename)
        if index==0:
            dharma_scas = xr.open_dataset(input_filename)
            dharma_scas['time'] = pd.to_datetime(dharma_scas['time'].values, unit='s', origin=pd.Timestamp(start_dtime))
            dharma_scas = dharma_scas.resample(time="600s",closed="right",label="right").mean()
        else:
            dharma_scas2 = xr.open_dataset(input_filename)
            dharma_scas2['time'] = pd.to_datetime(dharma_scas2['time'].values, unit='s', origin=pd.Timestamp(start_dtime))
            dharma_scas2 = dharma_scas2.resample(time="600s",closed="right",label="right").mean()
            dharma_scas = xr.concat([dharma_scas,dharma_scas2],dim='time')
    dharma_scas = dharma_scas.drop_duplicates('time',keep='first')
    #dharma_scas

    # calculate some additional variables requested
    dummy_sca = dharma_scas['lwp']*0.
    dharma_scas = dharma_scas.assign(Psurf = dummy_sca + dharma_params['sounding'].Psurf*100.)
    if do_ice:
        dharma_scas = dharma_scas.assign(opd_tot = dummy_sca + dharma_scas['opd_drops'].data + dharma_scas['opd_ice'].data)
    else:
        dharma_scas = dharma_scas.assign(opd_tot = dummy_sca + dharma_scas['opd_drops'].data)
        #dharma_scas = dharma_scas.assign(RHI = np.nan*dharma_scas['opd_tot'])
    dharma_scas = dharma_scas.assign(LWdnSFC = dummy_sca + dharma_snds['Flw_dn'].data[:,0])
    dharma_scas = dharma_scas.assign(LWupSFC = dummy_sca + dharma_snds['Flw_up'].data[:,0])
    dharma_scas = dharma_scas.assign(avg_precip_ice = dummy_sca + dharma_scas['avg_precip'].data 
                                 - dharma_snds['PFqc'].data[:,0] - dharma_snds['PFqr'].data[:,0])
    if do_progNa:
        dharma_scas = dharma_scas.assign(ssaf = dummy_sca + ssa_sfc)
    
    if verbose:
        dharma_scas

    # prepare output file in DEPHY format
    ####################################################### 

    # read list of requested variables
    vars_mean_list = pd.read_excel('output_conversion.xlsx',sheet_name='Mean')

    pd.set_option('display.max_rows', None)
    vars_mean_list

    # match DHARMA variables to requested outputs
    ####################################################### 
    # drop comments
    vars_mean_list = vars_mean_list#.drop(columns='comment (10-min average reported at endpoints, green=minimum)')

    # add columns to contain model output name and units conversion factors
    vars_mean_list = vars_mean_list.assign(model_name='missing data',conv_factor=1.0)

    # identify requested variables with only time dimension
    vars_mean_scas = vars_mean_list[vars_mean_list['dimensions']=='time']

    # match to DHARMA variable names and specify conversion factors
    for index in vars_mean_scas.index:
        standard_name = vars_mean_list.standard_name.iat[index]
        if standard_name=='surface_pressure': 
            vars_mean_list.model_name.iat[index] = 'Psurf'
        if standard_name=='surface_temperature': 
            vars_mean_list.model_name.iat[index] = 'avg_T_sfc'
        if standard_name=='surface_friction_velocity': 
            vars_mean_list.model_name.iat[index] = 'avg_ustar'
        if standard_name=='surface_roughness_length_for_momentum_in_air':
            vars_mean_list.model_name.iat[index] = 'avg_z0'
        if standard_name=='surface_roughness_length_for_heat_in_air':
            vars_mean_list.model_name.iat[index] = 'avg_z0h'
        if standard_name=='surface_roughness_length_for_humidity_in_air':
            # same as roughness length for heat in DHARMA
            vars_mean_list.model_name.iat[index] = 'avg_z0h'
        if standard_name=='surface_upward_sensible_heat_flux': 
            vars_mean_list.model_name.iat[index] = 'avg_T_flx'
        if standard_name=='surface_upward_latent_heat_flux': 
            vars_mean_list.model_name.iat[index] = 'avg_qv_flx'
        if standard_name=='obukhov_length': 
            vars_mean_list.model_name.iat[index] = 'avg_obk'
        if standard_name=='atmosphere_mass_content_of_liquid_cloud_water': 
            vars_mean_list.model_name.iat[index] = 'cwp'
            vars_mean_list.conv_factor.iat[index] = 1/1000.
        if standard_name=='atmosphere_mass_content_of_rain_water': 
            vars_mean_list.model_name.iat[index] = 'rwp'
            vars_mean_list.conv_factor.iat[index] = 1/1000.
        if do_ice:
            if standard_name=='atmosphere_mass_content_of_ice_water': 
                vars_mean_list.model_name.iat[index] = 'iwp'
                vars_mean_list.conv_factor.iat[index] = 1/1000.
        if standard_name=='cloud_area_fraction': 
            vars_mean_list.model_name.iat[index] = 'colf_opd'
        if standard_name=='optical_depth': 
            vars_mean_list.model_name.iat[index] = 'opd_tot'
        if standard_name=='optical_depth_of_liquid_cloud': 
            vars_mean_list.model_name.iat[index] = 'opd_cloud'
        if standard_name=='precipitation_flux_at_surface': 
            vars_mean_list.model_name.iat[index] = 'avg_precip'
            vars_mean_list.conv_factor.iat[index] = 1/3600.
        if do_ice:
            if standard_name=='precipitation_flux_at_surface_in_ice_phase': 
                vars_mean_list.model_name.iat[index] = 'avg_precip_ice'
                vars_mean_list.conv_factor.iat[index] = 1/3600.
        if standard_name=='optical_depth_of_cloud_droplets': 
            vars_mean_list.model_name.iat[index] = 'opd_cloud'
        if standard_name=='toa_outgoing_longwave_flux': 
            vars_mean_list.model_name.iat[index] = 'LWupTOA'
        if standard_name=='surface_downwelling_longwave_flux': 
            vars_mean_list.model_name.iat[index] = 'LWdnSFC'  
        if standard_name=='surface_upwelling_longwave_flux': 
            vars_mean_list.model_name.iat[index] = 'LWupSFC'
        if do_progNa:
            if standard_name=='surface_sea_spray_number_flux': 
                vars_mean_list.model_name.iat[index] = 'ssaf'

    # identify requested variables with time and vertical dimensions
    vars_mean_snds = vars_mean_list[vars_mean_list['dimensions']=='time, height']

    # match to DHARMA variable names and specify conversion factors
    for index in vars_mean_snds.index:
        standard_name = vars_mean_list.standard_name.iat[index]
        if standard_name=='air_pressure': 
            vars_mean_list.model_name.iat[index] = 'pressure'
        if standard_name=='eastward_wind': 
            vars_mean_list.model_name.iat[index] = 'u'
        if standard_name=='northward_wind': 
            vars_mean_list.model_name.iat[index] = 'v'
        if standard_name=='air_dry_density': 
            vars_mean_list.model_name.iat[index] = 'rhobar'
        if standard_name=='air_temperature': 
            vars_mean_list.model_name.iat[index] = 'T'
        if standard_name=='water_vapor_mixing_ratio': 
            vars_mean_list.model_name.iat[index] = 'qv'
        if standard_name=='relative_humidity': 
            vars_mean_list.model_name.iat[index] = 'RH'
            vars_mean_list.conv_factor.iat[index] = 1/100.
        if standard_name=='relative_humidity_over_ice': 
            vars_mean_list.model_name.iat[index] = 'RHI'
            vars_mean_list.conv_factor.iat[index] = 1/100.
        if standard_name=='air_potential_temperature': 
            vars_mean_list.model_name.iat[index] = 'theta'
        if standard_name=='specific_turbulent_kinetic_energy_resolved': 
            vars_mean_list.model_name.iat[index] = 'tkeavg'
        if standard_name=='specific_turbulent_kinetic_energy_sgs': 
            vars_mean_list.model_name.iat[index] = 'tke_smag'
        if standard_name=='mass_mixing_ratio_of_cloud_liquid_water_in_air': 
            vars_mean_list.model_name.iat[index] = 'qc'
        if standard_name=='mass_mixing_ratio_of_rain_water_in_air': 
            vars_mean_list.model_name.iat[index] = 'qr'
        if do_ice:
            if standard_name=='mass_mixing_ratio_of_cloud_ice_in_air': 
                vars_mean_list.model_name.iat[index] = 'qic'
            if standard_name=='mass_mixing_ratio_of_snow_in_air': 
                vars_mean_list.model_name.iat[index] = 'qif'
            if standard_name=='mass_mixing_ratio_of_graupel_in_air': 
                vars_mean_list.model_name.iat[index] = 'qid'
        if standard_name=='number_of_liquid_cloud_droplets_in_air': 
            vars_mean_list.model_name.iat[index] = 'nc'
        if standard_name=='number_of_rain_drops_in_air': 
            vars_mean_list.model_name.iat[index] = 'nr'
        if do_ice:
            if standard_name=='number_of_cloud_ice_crystals_in_air': 
                vars_mean_list.model_name.iat[index] = 'nic'
            if standard_name=='number_of_snow_crystals_in_air': 
                vars_mean_list.model_name.iat[index] = 'nif'
            if standard_name=='number_of_graupel_crystals_in_air': 
                vars_mean_list.model_name.iat[index] = 'nid'    
        if do_progNa:
            if standard_name=='number_of_total_aerosol_mode1': 
                vars_mean_list.model_name.iat[index] = 'na_1'
            if standard_name=='number_of_total_aerosol_mode2': 
                vars_mean_list.model_name.iat[index] = 'na_2'
            if standard_name=='number_of_total_aerosol_mode3': 
                vars_mean_list.model_name.iat[index] = 'na_3'
        if standard_name=='number_of_liquid_cloud_droplets_in_cloud': 
            vars_mean_list.model_name.iat[index] = 'nlcic'
        if do_ice:
            if standard_name=='number_of_ice_crystals_in_cloud': 
                vars_mean_list.model_name.iat[index] = 'niic'
        if standard_name=='dissipation_rate_of_turbulent_kinetic_energy': 
            vars_mean_list.model_name.iat[index] = 'eps'
        if standard_name=='zonal_momentum_flux': 
            vars_mean_list.model_name.iat[index] = 'uw_zt'
        if standard_name=='meridional_momentum_flux': 
            vars_mean_list.model_name.iat[index] = 'vw_zt'
        if standard_name=='variance_of_upward_air_velocity': 
            vars_mean_list.model_name.iat[index] = 'w2_zt'
        if standard_name=='vertical_flux_potential_temperature': 
            vars_mean_list.model_name.iat[index] = 'wth_zt'
        if standard_name=='vertical_flux_liquid_ice_water_potential_temperature': 
            vars_mean_list.model_name.iat[index] = 'wthli_zt'
        if standard_name=='vertical_flux_water_vapor': 
            vars_mean_list.model_name.iat[index] = 'wqv_zt'
        if standard_name=='vertical_flux_total_water': 
            vars_mean_list.model_name.iat[index] = 'wqt_zt'
        if standard_name=='area_fraction_of_liquid_cloud': 
            vars_mean_list.model_name.iat[index] = 'cloud_f'
        if standard_name=='precipitation_flux_in_air': 
            vars_mean_list.model_name.iat[index] = 'PF'
            vars_mean_list.conv_factor.iat[index] = 1/(3600.)
        if standard_name=='precipitation_flux_in_air_in_ice_phase': 
            vars_mean_list.model_name.iat[index] = 'PFi'
            vars_mean_list.conv_factor.iat[index] = 1/(3600.)
        if standard_name=='downwelling_longwave_flux_in_air': 
            vars_mean_list.model_name.iat[index] = 'LWdn'
        if standard_name=='upwelling_longwave_flux_in_air': 
            vars_mean_list.model_name.iat[index] = 'LWup'
        if standard_name=='tendency_of_air_potential_temperature_due_to_radiative_heating': 
            vars_mean_list.model_name.iat[index] = 'Srad'
            vars_mean_list.conv_factor.iat[index] = 1/3600.
        if standard_name=='tendency_of_air_potential_temperature_due_to_microphysics': 
            vars_mean_list.model_name.iat[index] = 'dth_micro'
            vars_mean_list.conv_factor.iat[index] = 1/3600.
        if standard_name=='tendency_of_air_potential_temperature_due_to_mixing': 
            vars_mean_list.model_name.iat[index] = 'dth_turb'
        if standard_name=='tendency_of_water_vapor_mixing_ratio_due_to_microphysics': 
            vars_mean_list.model_name.iat[index] = 'dq_micro'
            vars_mean_list.conv_factor.iat[index] = 1/3.6e6
        if standard_name=='tendency_of_water_vapor_mixing_ratio_due_to_mixing': 
            vars_mean_list.model_name.iat[index] = 'dq_turb'
        if do_progNa:
            if standard_name=='tendency_of_aerosol_number_due_to_warm_microphysics': 
                vars_mean_list.model_name.iat[index] = 'na_loss_liq'
                vars_mean_list.conv_factor.iat[index] = -1.
            if standard_name=='tendency_of_aerosol_number_due_to_mixing': 
                vars_mean_list.model_name.iat[index] = 'dna_mixing'
            if do_ice:
                if standard_name=='tendency_of_aerosol_number_due_to_cold_microphysics': 
                    vars_mean_list.model_name.iat[index] = 'na_loss_ice'
                    vars_mean_list.conv_factor.iat[index] = -1.
        if do_ice:
            if standard_name=='tendency_of_ice_number_due_to_heterogeneous_freezing': 
                vars_mean_list.model_name.iat[index] = 'Sice_het'
            if standard_name=='tendency_of_ice_number_due_to_secondary_ice_production': 
                vars_mean_list.model_name.iat[index] = 'Sice_sec'
            if standard_name=='tendency_of_ice_number_due_to_homogeneous_freezing': 
                vars_mean_list.model_name.iat[index] = 'Sice_hom'

    vars_mean_list[3:] # echo variables (first rows are dimensions)

    # create DEPHY output file
    ####################################################### 
    # create DEPHY output file
    dephy_filename = my_gitdir + my_outfile + '.nc'
    if os.path.exists(dephy_filename):
        os.remove(dephy_filename)
        print('The file ' + dephy_filename + ' has been deleted successfully') 
    dephy_file = Dataset(dephy_filename,mode='w',format='NETCDF3_CLASSIC')

    # create global attributes
    dephy_file.title='DHARMA LES results'
    dephy_file.reference='https://nasa-giss.github.io/LES-SCM/'
    dephy_file.authors='Ann Fridlind (ann.fridlind@nasa.gov) and Florian Tornow (florian.tornow@nasa.gov)'
    dephy_file.source=input_filename
    dephy_file.version=dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    dephy_file.format_version='DEPHY SCM format version 1.6'
    dephy_file.script='convert_DHARMA_LES_output_to_dephy_format.ipynb'
    dephy_file.startDate=start_dtime
    dephy_file.force_geo=1
    dephy_file.surfaceType='ocean (after spin-up)'
    dephy_file.surfaceForcing='ts (after spin-up)'
    dephy_file.lat=str(dharma_params['Coriolis'].lat) + ' deg N'
    dephy_file.dx=str(dharma_params['geometry'].L_x/dharma_params['geometry'].nx) + ' m'
    dephy_file.dy=str(dharma_params['geometry'].L_y/dharma_params['geometry'].ny) + ' m'
    dephy_file.dz='see zf variable'
    dephy_file.nx=dharma_params['geometry'].nx
    dephy_file.ny=dharma_params['geometry'].ny
    dephy_file.nz=dharma_params['geometry'].nz

    # create dimensions
    nz = dharma_snds.sizes['zt']
    zf = dephy_file.createDimension('zf', nz)
    zf = dephy_file.createVariable('zf', np.float64, ('zf',))
    zf.units = 'm'
    zf.long_name = 'height'
    zf[:] = dharma_snds['zt'].data

    ze = dephy_file.createDimension('ze', nz)
    ze = dephy_file.createVariable('ze', np.float64, ('ze',))
    ze.units = 'm'
    ze.long_name = 'layer_top_height'
    ze[:] = dharma_snds['zw'].data[1:]

    nt = dharma_snds.sizes['time']
    time = dephy_file.createDimension('time', nt)
    time = dephy_file.createVariable('time', np.float64, ('time',))
    time.units = 'seconds since ' + dephy_file.startDate
    time.long_name = 'time'
    # find time step and build time in seconds
    delta_t = (dharma_snds['time'].data[1]-dharma_snds['time'].data[0])/np.timedelta64(1, "s")
    time[:] = np.arange(nt)*delta_t

    # create and fill variables
    for index in vars_mean_list.index[2:]:
        std_name = vars_mean_list.standard_name.iat[index]
    #   print(std_name) # debug
        var_name = vars_mean_list.variable_id.iat[index]
        mod_name = vars_mean_list.model_name.iat[index]
        c_factor = vars_mean_list.conv_factor.iat[index]
        if vars_mean_list.dimensions.iat[index]=='time':
            new_sca = dephy_file.createVariable(var_name, np.float64, ('time'))
            new_sca.units = vars_mean_list.units.iat[index]
            new_sca.long_name = std_name
            if vars_mean_list.model_name.iat[index]!='missing data' and mod_name in dharma_scas:
                if (var_name == 'z0'):
                    new_sca[:] = np.round(dharma_scas[mod_name].data*c_factor,4)
                elif (var_name == 'z0h')| (var_name == 'z0q'):
                    new_sca[:] = np.round(dharma_scas[mod_name].data*c_factor,7)
                else:
                    new_sca[:] = dharma_scas[mod_name].data*c_factor
        if vars_mean_list.dimensions.iat[index]=='time, height':
            new_snd = dephy_file.createVariable(var_name, np.float64, ('time','zf'))
            new_snd.units = vars_mean_list.units.iat[index]
            new_snd.long_name = std_name
            if vars_mean_list.model_name.iat[index]!='missing data' and mod_name in dharma_snds: 
                new_snd[:] = dharma_snds[mod_name].data*c_factor

    
    if verbose:
        print(dephy_file)
    dephy_file.close()

    # check output file
    ####################################################### 

    if verbose:
        dephy_check = xr.open_dataset(dephy_filename)
        dephy_check

    print('--------------------')
    print(' ')