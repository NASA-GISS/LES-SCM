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


def ModelE_convert(path,output_filename,verbose=False):

    ## translated from COMBLE notebook written by Ann Fridlind (https://github.com/ARM-Development/comble-   mip/blob/main/notebooks/conversion_output/convert_DHARMA_LES_output_to_dephy_format.ipynb)

    print('--------------------')
    
    # specify start time of simulation and simulation name
    ######################################################
    if '20200313' in output_filename:
        start_dtime = '2020-03-12 22:00:00.0'
    
    # specify simulation
    my_input_suffix = path             #*.nc
    my_output_suffix = output_filename #*.nc

    my_input_dir = '/discover/nobackup/tflorian/modelE_runs/'

    # specify Github scratch directory where processed model output will be committed
    my_output_filename = my_output_suffix
    my_gitdir = '/home/tflorian/LES-SCM/output_scm/modele/'
    
    # Read single file containing all output data
    input_filename = my_input_dir + path + '/allsteps.allmerge' + my_input_suffix + '.nc'
    print(input_filename)
    modele_data = xr.open_dataset(input_filename)

    # check if the run contains ice variables
    do_ice = bool(max(modele_data['iwp'].values)>0.)
    print('do_ice = ',do_ice)

    # full parameter list
    modele_data

    #Calculate and append additional variables
    dummy_sca = modele_data['lwp']*0.
    modele_data = modele_data.assign(clwpt = dummy_sca + modele_data['cLWPss'].data + modele_data['cLWPmc'].data)
    modele_data = modele_data.assign(rlwpt = dummy_sca + modele_data['pLWPss'].data + modele_data['pLWPmc'].data)
    if do_ice: modele_data = modele_data.assign(iwpt = dummy_sca + modele_data['cIWPss'].data + modele_data['cIWPmc'].data + 
                                                modele_data['pIWPss'].data + modele_data['pIWPmc'].data)
    modele_data = modele_data.assign(tau = dummy_sca + modele_data['tau_ss'].data + modele_data['tau_mc'].data)
    
    dummy_snd = modele_data['q']*0.
    modele_data = modele_data.assign(rhobar = dummy_snd + 100.*modele_data['p_3d'].data/(287.05*modele_data['t'].data))
    modele_data = modele_data.assign(qlc = dummy_snd + modele_data['qcl'].data + modele_data['QCLmc'].data)
    modele_data = modele_data.assign(qlr = dummy_snd + modele_data['qpl'].data + modele_data['QPLmc'].data)
    if do_ice: modele_data = modele_data.assign(qi = dummy_snd + modele_data['qci'].data + modele_data['qpi'].data + 
                                                modele_data['QCImc'].data + modele_data['QPImc'].data)
    modele_data = modele_data.assign(lcf = dummy_snd + modele_data['cldsscl'].data + modele_data['cldmccl'].data)
    modele_data['lcf'].values = np.clip(modele_data['lcf'].values,0.,1.)
    modele_data = modele_data.assign(prt = dummy_snd + modele_data['ssp_cl_3d'].data + modele_data['ssp_pl_3d'].data + modele_data['rain_mc'].data)
    if do_ice: 
        modele_data = modele_data.assign(pit = dummy_snd + modele_data['ssp_ci_3d'].data + modele_data['ssp_pi_3d'].data + modele_data['snow_mc'].data)
        modele_data['prt'].data += modele_data['pit'].data
    modele_data = modele_data.assign(dth_micro = dummy_snd + modele_data['dth_ss'].data + modele_data['dth_mc'].data)
    modele_data = modele_data.assign(dq_micro = dummy_snd + modele_data['dq_ss'].data + modele_data['dq_mc'].data)
    modele_data = modele_data.assign(wqt_turb = dummy_snd + modele_data['wq_turb'].data + modele_data['wql_turb'].data + modele_data['wqi_turb'].data)

    #Prepare output file in DEPHY format
    # read list of requested variables
    
    vars_mean_list = pd.read_excel('output_conversion_scm.xlsx',sheet_name='SCM')
    pd.set_option('display.max_rows', None)
    vars_mean_list

    # Match ModelE3 variables to requested outputs
    # drop comments
    vars_mean_list = vars_mean_list.drop(columns='comment (reported at end of each model physics time step, green=minimum, red=granularity enabling EMC2)')

    # add columns to contain model output name and units conversion factors
    vars_mean_list = vars_mean_list.assign(model_name='missing data',conv_factor=1.0)

    # match to ModelE3 variable names and specify conversion factors
    for index in vars_mean_list.index:
        standard_name = vars_mean_list.standard_name.iat[index]
        if standard_name=='air_pressure': 
            vars_mean_list.model_name.iat[index] = 'p_3d'
    #    if standard_name=='layer_top_pressure': 
    #        vars_mean_list.model_name.iat[index] = 'pe_t'
        if standard_name=='surface_pressure': 
            vars_mean_list.model_name.iat[index] = 'prsurf'
            vars_mean_list.conv_factor.iat[index] = 100.
        if standard_name=='surface_temperature': 
            vars_mean_list.model_name.iat[index] = 'gtempr'
        if standard_name=='surface_friction_velocity': 
            vars_mean_list.model_name.iat[index] = 'ustar'
    #    if standard_name=='surface_roughness_length_for_momentum_in_air': 
    #        vars_mean_list.model_name.iat[index] = 'z0m'
    #    if standard_name=='surface_roughness_length_for_heat_in_air': 
    #        vars_mean_list.model_name.iat[index] = 'z0h'
    #    if standard_name=='surface_roughness_length_for_humidity_in_air': 
    #        vars_mean_list.model_name.iat[index] = 'z0q'
        if standard_name=='surface_upward_sensible_heat_flux': 
            vars_mean_list.model_name.iat[index] = 'shflx'
            vars_mean_list.conv_factor.iat[index] = -1.
        if standard_name=='surface_upward_latent_heat_flux': 
            vars_mean_list.model_name.iat[index] = 'lhflx'
            vars_mean_list.conv_factor.iat[index] = -1.
        if standard_name=='obukhov_length': 
            vars_mean_list.model_name.iat[index] = 'lmonin'
        if standard_name=='pbl_height': 
            vars_mean_list.model_name.iat[index] = 'pblht_bp'
        if standard_name=='inversion_height': 
            vars_mean_list.model_name.iat[index] = 'pblht_th'
        if standard_name=='atmosphere_mass_content_of_liquid_cloud_water': 
            vars_mean_list.model_name.iat[index] = 'clwpt'
            vars_mean_list.conv_factor.iat[index] = 0.001
        if standard_name=='atmosphere_mass_content_of_rain_water': 
            vars_mean_list.model_name.iat[index] = 'rlwpt'
            vars_mean_list.conv_factor.iat[index] = 0.001
        if do_ice:
            if standard_name=='atmosphere_mass_content_of_ice_water': 
                vars_mean_list.model_name.iat[index] = 'iwpt'
                vars_mean_list.conv_factor.iat[index] = 0.001
        if standard_name=='area_fraction_cover_of_hydrometeors': 
            vars_mean_list.model_name.iat[index] = 'cldtot_2d'
    #    if standard_name=='area_fraction_cover_of_liquid_cloud': 
    #        vars_mean_list.model_name.iat[index] = ''
        if standard_name=='area_fraction_cover_of_convective_hydrometeors': 
            vars_mean_list.model_name.iat[index] = 'cldmc_2d'
        if standard_name=='optical_depth': 
            vars_mean_list.model_name.iat[index] = 'tau'
    #    if standard_name=='optical_depth_of_liquid_water': 
    #        vars_mean_list.model_name.iat[index] = ''
        if standard_name=='precipitation_flux_at_surface': 
            vars_mean_list.model_name.iat[index] = 'prec'
            vars_mean_list.conv_factor.iat[index] = 1./86400
        if standard_name=='precipitation_flux_of_ice_at_surface': 
            vars_mean_list.model_name.iat[index] = 'snowfall'
            vars_mean_list.conv_factor.iat[index] = 1./86400
        if standard_name=='toa_outgoing_longwave_flux': 
            vars_mean_list.model_name.iat[index] = 'olr'
        if standard_name=='surface_downwelling_longwave_flux': 
            vars_mean_list.model_name.iat[index] = 'lwds'  
        if standard_name=='surface_upwelling_longwave_flux': 
            vars_mean_list.model_name.iat[index] = 'lwus'  
        if standard_name=='height': 
            vars_mean_list.model_name.iat[index] = 'z'
        if standard_name=='eastward_wind': 
            vars_mean_list.model_name.iat[index] = 'u'
        if standard_name=='northward_wind': 
            vars_mean_list.model_name.iat[index] = 'v'
        if standard_name=='air_dry_density': 
            vars_mean_list.model_name.iat[index] = 'rhobar'
        if standard_name=='air_temperature': 
            vars_mean_list.model_name.iat[index] = 't'
        if standard_name=='water_vapor_mixing_ratio': 
            vars_mean_list.model_name.iat[index] = 'q'
        if standard_name=='relative_humidity': 
            vars_mean_list.model_name.iat[index] = 'rhw'
            vars_mean_list.conv_factor.iat[index] = 0.01
        if standard_name=='relative_humidity_over_ice': 
            vars_mean_list.model_name.iat[index] = 'rhi'
            vars_mean_list.conv_factor.iat[index] = 0.01
        if standard_name=='air_potential_temperature': 
            vars_mean_list.model_name.iat[index] = 'th'
        if standard_name=='mass_mixing_ratio_of_cloud_liquid_water_in_air': 
            vars_mean_list.model_name.iat[index] = 'qlc'
        if standard_name=='mass_mixing_ratio_of_rain_water_in_air': 
            vars_mean_list.model_name.iat[index] = 'qlr'
        if do_ice: 
            if standard_name=='mass_mixing_ratio_of_ice_water_in_air': 
                vars_mean_list.model_name.iat[index] = 'qi'
        if standard_name=='area_fraction_of_hydrometeors': 
            vars_mean_list.model_name.iat[index] = 'cfr'
        if standard_name=='area_fraction_of_liquid_cloud': 
            vars_mean_list.model_name.iat[index] = 'lcf'
        if standard_name=='area_fraction_of_convective_hydrometeors': 
            vars_mean_list.model_name.iat[index] = 'cldmcr'
        if standard_name=='precipitation_flux_in_air': 
            vars_mean_list.model_name.iat[index] = 'prt'
            vars_mean_list.conv_factor.iat[index] = 1./86400
        if do_ice:
            if standard_name=='precipitation_flux_in_air_in_ice_phase': 
                vars_mean_list.model_name.iat[index] = 'pit'
                vars_mean_list.conv_factor.iat[index] = 1./86400
        if standard_name=='specific_turbulent_kinetic_energy': 
            vars_mean_list.model_name.iat[index] = 'e_turb'
        if standard_name=='disspation_rate_of_turbulent_kinetic_energy': 
            vars_mean_list.model_name.iat[index] = 'dissip_tke_turb'
            vars_mean_list.conv_factor.iat[index] = -1.
        if standard_name=='zonal_momentum_flux': 
            vars_mean_list.model_name.iat[index] = 'uw_turb'
        if standard_name=='meridional_momentum_flux': 
            vars_mean_list.model_name.iat[index] = 'vw_turb'
        if standard_name=='variance_of_upward_air_velocity': 
            vars_mean_list.model_name.iat[index] = 'w2_turb'
        if standard_name=='vertical_flux_potential_temperature': 
            vars_mean_list.model_name.iat[index] = 'wth_turb'
    #    if standard_name=='vertical_flux_liquid_water_potential_temperature': 
    #        vars_mean_list.model_name.iat[index] = ''
        if standard_name=='vertical_flux_water_vapor': 
            vars_mean_list.model_name.iat[index] = 'wq_turb'
        if standard_name=='vertical_flux_total_water': 
            vars_mean_list.model_name.iat[index] = 'wqt_turb'
        if standard_name=='convection_updraft_mass_flux': 
            vars_mean_list.model_name.iat[index] = 'lwdp'
        if standard_name=='convection_downdraft_mass_flux': 
            vars_mean_list.model_name.iat[index] = 'lwdp'
        if standard_name=='downwelling_longwave_flux_in_air': 
            vars_mean_list.model_name.iat[index] = 'lwdp'
        if standard_name=='upwelling_longwave_flux_in_air': 
            vars_mean_list.model_name.iat[index] = 'lwup'
        if standard_name=='tendency_of_air_potential_temperature_due_to_radiative_heating': 
            vars_mean_list.model_name.iat[index] = 'dth_lw'
            vars_mean_list.conv_factor.iat[index] = 1./86400
        if standard_name=='tendency_of_air_potential_temperature_due_to_microphysics': 
            vars_mean_list.model_name.iat[index] = 'dth_micro'
            vars_mean_list.conv_factor.iat[index] = 1./86400
        if standard_name=='tendency_of_air_potential_temperature_due_to_mixing': 
            vars_mean_list.model_name.iat[index] = 'dth_turb'
            vars_mean_list.conv_factor.iat[index] = 1./86400
        if standard_name=='tendency_of_water_vapor_mixing_ratio_due_to_microphysics': 
            vars_mean_list.model_name.iat[index] = 'dq_micro'
            vars_mean_list.conv_factor.iat[index] = 1./86400
        if standard_name=='tendency_of_water_vapor_mixing_ratio_due_to_mixing': 
            vars_mean_list.model_name.iat[index] = 'dq_turb'
            vars_mean_list.conv_factor.iat[index] = 1./86400
        if standard_name=='mass_mixing_ratio_of_liquid_cloud_water_in_air_stratiform': 
            vars_mean_list.model_name.iat[index] = 'qcl'
        if standard_name=='mass_mixing_ratio_of_rain_water_in_air_stratiform': 
            vars_mean_list.model_name.iat[index] = 'qpl'
        if do_ice:
            if standard_name=='mass_mixing_ratio_of_ice_cloud_in_air_stratiform': 
                vars_mean_list.model_name.iat[index] = 'qci'
            if standard_name=='mass_mixing_ratio_of_ice_precipitation_in_air_stratiform': 
                vars_mean_list.model_name.iat[index] = 'qpi'
        if standard_name=='mass_mixing_ratio_of_liquid_cloud_water_in_air_convective': 
            vars_mean_list.model_name.iat[index] = 'QCLmc'
        if standard_name=='mass_mixing_ratio_of_rain_water_in_air_convective': 
            vars_mean_list.model_name.iat[index] = 'QPLmc'
        if do_ice:
            if standard_name=='mass_mixing_ratio_of_ice_cloud_in_air_convective': 
                vars_mean_list.model_name.iat[index] = 'QCImc'
            if standard_name=='mass_mixing_ratio_of_ice_precipitation_in_air_convective': 
                vars_mean_list.model_name.iat[index] = 'QPImc'
        if standard_name=='number_of_liquid_cloud_droplets_in_air_stratiform': 
            vars_mean_list.model_name.iat[index] = 'ncl'
        if standard_name=='number_of_rain_drops_in_air_stratiform': 
            vars_mean_list.model_name.iat[index] = 'npl'
        if do_ice:
            if standard_name=='number_of_ice_cloud_crystals_in_air_stratiform': 
                vars_mean_list.model_name.iat[index] = 'nci'
            if standard_name=='number_of_ice_precipitation_crystals_in_air_stratiform': 
                vars_mean_list.model_name.iat[index] = 'npi'
        if standard_name=='effective_radius_of_liquid_cloud_droplets_convective': 
            vars_mean_list.model_name.iat[index] = 're_mccl'
        if standard_name=='effective_radius_of_rain_convective': 
            vars_mean_list.model_name.iat[index] = 're_mcpl'
        if do_ice:
            if standard_name=='effective_radius_of_ice_cloud_convective': 
                vars_mean_list.model_name.iat[index] = 're_mcci'
            if standard_name=='effective_radius_of_ice_precipitation_convective': 
                vars_mean_list.model_name.iat[index] = 're_mcpi'
        if standard_name=='area_fraction_of_liquid_cloud_stratiform': 
            vars_mean_list.model_name.iat[index] = 'cldsscl'
        if standard_name=='area_fraction_of_rain_stratiform': 
            vars_mean_list.model_name.iat[index] = 'cldsspl'
        if do_ice:
            if standard_name=='area_fraction_of_ice_cloud_stratiform': 
                vars_mean_list.model_name.iat[index] = 'cldssci'
            if standard_name=='area_fraction_of_ice_precipitation_stratiform': 
                vars_mean_list.model_name.iat[index] = 'cldsspi'
        if standard_name=='area_fraction_of_liquid_cloud_convective': 
            vars_mean_list.model_name.iat[index] = 'cldmccl'
        if standard_name=='area_fraction_of_rain_convective': 
            vars_mean_list.model_name.iat[index] = 'cldmcpl'
        if do_ice:
            if standard_name=='area_fraction_of_ice_cloud_convective': 
                vars_mean_list.model_name.iat[index] = 'cldmcci'
            if standard_name=='area_fraction_of_ice_precipitation_convective': 
                vars_mean_list.model_name.iat[index] = 'cldmcpi'
        if standard_name=='mass_weighted_fall_speed_of_liquid_cloud_water_stratiform': 
            vars_mean_list.model_name.iat[index] = 'vm_sscl'
        if standard_name=='mass_weighted_fall_speed_of_rain_stratiform': 
            vars_mean_list.model_name.iat[index] = 'vm_sspl'
        if do_ice:
            if standard_name=='mass_weighted_fall_speed_of_ice_cloud_stratiform': 
                vars_mean_list.model_name.iat[index] = 'vm_ssci'
            if standard_name=='mass_weighted_fall_speed_of_ice_precipitation_stratiform': 
                vars_mean_list.model_name.iat[index] = 'vm_sspi'
        if standard_name=='mass_weighted_fall_speed_of_liquid_cloud_water_convective': 
            vars_mean_list.model_name.iat[index] = 'vm_mccl'
        if standard_name=='mass_weighted_fall_speed_of_rain_convective': 
            vars_mean_list.model_name.iat[index] = 'vm_mcpl'
        if do_ice:
            if standard_name=='mass_weighted_fall_speed_of_cloud_ice_crystals_convective': 
                vars_mean_list.model_name.iat[index] = 'vm_mcci'
            if standard_name=='mass_weighted_fall_speed_of_ice_precipitation_convective': 
                vars_mean_list.model_name.iat[index] = 'vm_mcpi'

    vars_mean_list[2:] # echo variables (first two rows are dimensions)
    # Create DEPHY output file
    dephy_filename =  my_gitdir + my_output_filename + '.nc'
    if os.path.exists(dephy_filename):
        os.remove(dephy_filename)
        print('The file ' + dephy_filename + ' has been deleted successfully')    
    dephy_file = Dataset(dephy_filename,mode='w',format='NETCDF3_CLASSIC')
    start_date = '2020-03-12T22:00:00Z'

    # create global attributes
    dephy_file.title='ModelE3 SCM results for COMBLE-MIP case: fixed stratiform Nd and Ni'
    dephy_file.reference='https://github.com/ARM-Development/comble-mip'
    dephy_file.authors='Ann Fridlind (ann.fridlind@nasa.gov), Florian Tornow (florian.tornow@nasa.gov), Andrew Ackerman (andrew.ackerman@nasa.gov)'
    dephy_file.source=input_filename
    dephy_file.version=dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    dephy_file.format_version='DEPHY SCM format version 1.6'
    dephy_file.script='convert_ModelE3_SCM_output_to_dephy_format.ipynb'
    dephy_file.startDate=start_date
    dephy_file.force_geo=1
    dephy_file.surfaceType='ocean'
    dephy_file.surfaceForcing='ts'
    dephy_file.lat='74.5 deg N'
    dephy_file.dp='see pressure variable'
    dephy_file.np=modele_data.sizes['p']

    # create dimensions
    nt = modele_data.sizes['time']
    time = dephy_file.createDimension('time', nt)
    time = dephy_file.createVariable('time', np.float64, ('time',))
    time.units = 'seconds since ' + dephy_file.startDate
    time.long_name = 'time'
    # find time step and build time in seconds
    time1 = dt.datetime.strptime(str(modele_data['time'].data[0]),'%Y-%m-%d %H:%M:%S')
    time2 = dt.datetime.strptime(str(modele_data['time'].data[1]),'%Y-%m-%d %H:%M:%S')
    delta_t = (time2-time1).total_seconds()
    time[:] = (np.arange(nt)+1.)*delta_t

    nl = modele_data.sizes['p']
    layer = dephy_file.createDimension('layer', nl)
    layer = dephy_file.createVariable('layer', np.float64, ('layer',))
    layer.units = '1'
    layer.long_name = 'pressure_layer'
    layer[:] = np.arange(nl)+1

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
            if vars_mean_list.model_name.iat[index]!='missing data':
                new_sca[:] = modele_data[mod_name].data*c_factor
        if vars_mean_list.dimensions.iat[index]=='time, layer':
            new_snd = dephy_file.createVariable(var_name, np.float64, ('time','layer'))
            new_snd.units = vars_mean_list.units.iat[index]
            new_snd.long_name = std_name
            if vars_mean_list.model_name.iat[index]!='missing data': 
                new_snd[:] = modele_data[mod_name].data*c_factor

    print(dephy_file)
    dephy_file.close()

    # Check output file
    dephy_check = xr.open_dataset(dephy_filename)
    dephy_check
