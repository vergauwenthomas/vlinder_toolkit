#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 12:01:35 2022

@author: thoverga
"""
import os
from pathlib import Path
main_folder = Path(__file__).resolve().parents[1]
testdata_file = os.path.join(str(main_folder), 'tests', 'test_data',  'vlinderdata_small.csv' )
# metadata = os.path.join(str(main_folder), 'static_data', 'vlinder_metadata.csv')


import vlinder_toolkit



#%%

# =============================================================================
# Settings
# =============================================================================


# 1. Initiate settings object. This object contains all settings needed for furthur analysis
settings = vlinder_toolkit.Settings()




# 3. If the output data folder and input file are not exported as system variables, you need to update them:
settings.update_settings(input_data_file=testdata_file, #A demo data file, downloaded with brian tool: https://vlinder.ugent.be/vlinderdata/multiple_vlinders.php
                         output_folder='/home/$USER/output/')





# 4. Check the setting by using the .show() or .check_settings() function 
settings.show()





# =============================================================================
#  Import data
# =============================================================================



#1. Importing a dataset containing mulitple different stations is a function in the Dataset class. First we need to initiate a Dataset with a name of choise.

sept_2022_all_vlinders = vlinder_toolkit.Dataset()


# ---------------- Importing from CSV file -------------------------------


#The dataset is initiated but still empty. Filling it with the data from a csv file is simply done by:
    
sept_2022_all_vlinders.import_data_from_file(settings) #Rember that you added the input file in the settings object, this file will be used.


# =============================================================================
# Applying quality control
# =============================================================================

#  a number of quality control methods are available in the toolkit. We can classify them in two groups:
    # 1) Quality control for missing/duplicated timestamps 
    
        #These are automatically performed when the dataset is created
        
    # 2) Quality control for bad observations
    
        #These are not automatically executed. These checks are performd in a sequence 
        # of specific checks, that are looking for signatures of typically bad observations.
        # The following checks are available:
            # Gross value check: a threshold check that observations should be between the thresholds
            # Persistance check: a check that looks for repetitive observation values (indicating a connection error.)
            # Spike check: Check if an observation does not change between two timestamps by more than a threshold.
            # Internal consistency check: Check if the temperature observations are physically valid based on relative humidity observations.
            
#All settings, labels, replacement values are defind in the default settings in /settings_files/qc_settings.py
#To inspect (and change) these quality control settings, you can extract them out of the Settings:

qc_settings = settings.qc_check_settings #Settings are stored in nested dictionary
print(qc_settings)

#Changing a setting example:
settings.qc_check_settings['persistance']['temp']['max_valid_repetitions'] = 6


        
# Quality control checks are always applied on the full dataset Using the apply_quality_control method on the dataset.

sept_2022_all_vlinders.apply_quality_control(obstype='temp', #which observations to check
                                             gross_value=True, #apply gross_value check?
                                             persistance=True, #apply persistance check?
                                             step=True, #apply the step chec?
                                             internal_consistency=True # apply internal consistency check?
                                             )

# =============================================================================
# Quality control values
# =============================================================================

# If an observation is flagged as an outlier by a check, the observational value is replaced.
# By default the outliers are replaced by Nan-values. You can see or change the default replacement values:

print(settings.qc_outlier_values)


# =============================================================================
# Quality control labels
# =============================================================================

# Each check makes a label for each observation. These labels are stored in the dataset.df:

print(sept_2022_all_vlinders.df.head())

# You can see that there is a specific coloumn for each check-label (and for each observationtype that is checked):
print(sept_2022_all_vlinders.df.columns)

# The labels:
#   ok: Observation is not flagged as an outlier    
#   not checked: observation could not be checked (mostly because the observations is already flagged as outlier by a previous check)
#   **** outlier: labeld as outlier by the **** check


# After applying the Quality control each observations (that is checked) has a set of qc-labels. 
# To create One column with the final label (based on the labels for each check), you can call the 
# add_final_qc_labels, which will add a final-qc-label column in the dataset.df:

sept_2022_all_vlinders.add_final_qc_labels()

print(sept_2022_all_vlinders.df['temp_final_label'].head())

# (When writing a dataset to file, there is an attribute 'add_final_labels'. When 
#  True, the final labels are computed and added to the file.)


# =============================================================================
# Quality control stats
# =============================================================================

#  Some basic overview statistics are implemented for the quality controll labels. 
#  You can extract the frequency statistics by calling:
    
    # obstype: Get statistics of the QC applied on which observationtype
    # stationnames: Which subset of the dataset to use? If None, statistics are computed over all the stations,
    #               if one stationname is given the stats are computed for that station. You can also give a list of stationnames. 
    # make_plot:  If True, an overview of pie-charts per check are made.
    
    #The output is a dataframe with frequency statistics per check PRESENTED IN PERSETNTAGES.
    
qc_statistics = sept_2022_all_vlinders.get_qc_stats(obstype='temp',
                                                    stationnames=None, #or 'station_A' or list of stationnames ['station_A', 'station_B']
                                                    make_plot=True)    

print(qc_statistics)

        


# =============================================================================
# Timeseries plots
# =============================================================================

#NOTE: All styling settings are defined in the src/vlinder_toolkit/settings_files/default_formats_settings.py


#To plot timeseries for one station you can use the make_plot function on the station object:
favorite_station = sept_2022_all_vlinders.get_station(stationname='vlinder05')


#Possible obstypes to plot:
    # 'temp','radiation_temp','humidity','precip','precip_sum','wind_speed',
    # 'wind_gust','wind_direction','pressure','pressure_at_sea_level'
    
favorite_station.make_plot(variable='temp', 
                            title=None) #if title=None, a title will be generated

# You can see the effect of the quality control where the values are replaced when the observation was flagged as outlier.
# Since Nan values are not shown in the plotting, the effect of the QC can be seen by missing observations and gaps.





#If you want to compair multiple stations by a timeseries you can use the make_plot function on the dataset:

from datetime import datetime    

sept_2022_all_vlinders.make_plot(stationnames=['vlinder02', 'vlinder05', 'vlinder07'],
                                variable='humidity',
                                starttime=datetime(2022, 9,4), # 2022/09/04 00:00:00
                                endtime=datetime(2022,9,12,12,45), #2022/09/12 12:45:00
                                title=None,
                                legend=True
                                )
#If you do not specify a start and endtime, all available timestamps are used.
















