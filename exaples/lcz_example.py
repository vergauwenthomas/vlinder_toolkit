#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 12:01:35 2022

@author: thoverga
"""
import os
from pathlib import Path
main_folder = Path(__file__).resolve().parents[1]
testdata_file = os.path.join(str(main_folder), 'tests', 'test_data',  'vlinderdata.csv' )


import vlinder_toolkit



#%%

# =============================================================================
# Settings
# =============================================================================


# 1. Initiate settings object. This object contains all settings needed for furthur analysis
settings = vlinder_toolkit.Settings()



# 2. To use the LCZ functions, you need to add the location of the LCZ map to the settings.
# First you have to download the LCZ (world) map, this map is free available here:
    #  https://zenodo.org/record/6364594/files/lcz_filter_v1.tif?download=1
# products or published results have to refer the autors of this map:
    # Matthias Demuzere, Jonas Kittner, Alberto Martilli, Gerald Mills, Christian Moede, Iain D. Stewart, Jasper van Vliet, & Benjamin Bechtel. (2022). Global map of Local Climate Zones (1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.6364594

lcz_map_location = os.path.join(str(main_folder), 'physiograpy', 'lcz_filter_v1.tif')



# 3. If the output data folder and input file are not exported as system variables, you need to update them:
settings.update_settings(input_file=testdata_file, #A demo data file, downloaded with brian tool: https://vlinder.ugent.be/vlinderdata/multiple_vlinders.php
                         output_data_folder='/home/$USER/output/',
                         geotiff_lcz_file=lcz_map_location) #add lcz location to Settings.





# 4. Check the setting by using the .show() or .check_settings() function 
settings.show()





# =============================================================================
#  Import data
# =============================================================================



#1. Importing a dataset containing mulitple different stations is a function in the Dataset class. First we need to initiate a Dataset with a name of choise.

aug_2020_all_vlinders = vlinder_toolkit.Dataset()


# ---------------- Importing from CSV file -------------------------------


#The dataset is initiated but still empty. Filling it with the data from a csv file is simply done by:
    
aug_2020_all_vlinders.import_data_from_file(settings) #Rember that you added the input file in the settings object, this file will be used.


#You can see at any time what is in the dataset by:
aug_2020_all_vlinders.show()



# =============================================================================
# Analysing LCZ
# =============================================================================


#To get the lcz from a station you can use the get_lcz function on a station:
favorite_station = aug_2020_all_vlinders.get_station(stationname='vlinder02')

lcz = favorite_station.get_lcz()

print(lcz)






# To make a geospatial map of the LCZ of all stations:
aug_2020_all_vlinders.make_geo_plot(varible='lcz')










