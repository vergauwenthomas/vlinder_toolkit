#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The class object for a Vlinder/mocca station
@author: thoverga
"""
# from collections.abc import Iterable

import pandas as pd
import geopandas as gpd
import numpy as np
import os
from datetime import datetime
import logging

from .settings import Settings
from .data_import import import_data_from_csv, import_data_from_database, template_to_package_space, import_metadata_from_csv
from .data_import import coarsen_time_resolution
from .landcover_functions import geotiff_point_extraction
from .geometry_functions import find_largest_extent
from .plotting_functions import spatial_plot, timeseries_plot, timeseries_comp_plot
from .qc_checks import gross_value, persistance

logger = logging.getLogger(__name__)

# =============================================================================
# field classifiers
# =============================================================================

#Static fields are fields (attributes and observations) that do not change in time
static_fields = ['network', 'name', 
                'lat', 'lon', #TODO make these dynamic, now used as static 
                'call_name', 'location',
                'lcz']

#Categorical fields are fields with values that are assumed to be categorical.
#Note: (there are static and dynamic fields that are categorical)
categorical_fields = ['wind_direction', 'lcz']


observation_types = ['temp', 'radiation_temp', 'humidity', 'precip',
                     'precip_sum', 'wind_speed', 'wind_gust', 'wind_direction',
                     'pressure', 'pressure_at_sea_level']

location_info = ['name', 'network', 'lat', 'lon', 'call_name', 'location', 'lcz']

# =============================================================================
# station class
# =============================================================================
class Station:
    def __init__(self, station_name, network_name):
        logger.debug(f'Station {station_name} initialisation.')
        self.network = network_name
        self.name = station_name
        
        #Meta data without processing
        self.lat = pd.Series(dtype='float64', name='lat')
        self.lon = pd.Series(dtype='float64', name='lon')
        self.call_name = None #ex. Antwerpen Zoo
        self.location = None #ex. Antwerpen 
        
        #Observations
        self.temp = pd.Series(dtype='float64', name='temp')
        self.radiation_temp = pd.Series(dtype='float64', name='radiation_temp') 
        
        self.humidity = pd.Series(dtype='float64', name='humidity')
        
        self.precip = pd.Series(dtype='float64', name='precip')
        self.precip_sum = pd.Series(dtype='float64', name='precip_sum')
        
        self.wind_speed = pd.Series(dtype='float64', name='wind_speed')
        self.wind_gust = pd.Series(dtype='float64', name='wind_gust')
        self.wind_direction = pd.Series(dtype='float64', name='wind_direction')
        
        self.pressure = pd.Series(dtype='float64', name='pressure')
        self.pressure_at_sea_level = pd.Series(dtype='float64', name='pressure_at_sea_level')
        
        #physiographic data
        self.lcz = None
        
        #Units and descriptions
        self.units = {'temp': None,
                      'radiation_temp': None,
                      'humidity': None,
                      'precip': None,
                      'precip_sum': None,
                      'wind_speed': None,
                      'wind_gust': None,
                      'wind_direction': None,
                      'pressure': None,
                      'pressure_at_sea_level': None}
        self.obs_description = {'temp': None,
                          'radiation_temp': None,
                          'humidity': None,
                          'precip': None,
                          'precip_sum': None,
                          'wind_speed': None,
                          'wind_gust': None,
                          'wind_direction': None,
                          'pressure': None,
                          'pressure_at_sea_level': None}
        
        #attributes that can be filled with info from other functions
        self.qc_labels_df =  {'temp': pd.DataFrame(),
                              'radiation_temp': pd.DataFrame(),
                              'humidity': pd.DataFrame(),
                              'precip': pd.DataFrame(),
                              'precip_sum': pd.DataFrame(),
                              'wind_speed': pd.DataFrame(),
                              'wind_gust': pd.DataFrame(),
                              'wind_direction': pd.DataFrame(),
                              'pressure': pd.DataFrame(),
                              'pressure_at_sea_level': pd.DataFrame()}
        

    def show(self):
        logger.debug(f'Show {self.name} info.')
        print(' ------- Static info -------')
        print('Stationname: ', self.name)
        print('Network: ', self.network)
        print('Call name: ', self.call_name)
        print('Location: ', self.location)
        print('latitude: ', self.lat)
        print('longtitude: ',self.lon)
       
        print(' ------- Physiography info -------')
        print('LCZ: ', self.lcz)
        print(' ')
        print(' ------- Observations info -------')

        
        if self.df().empty:
            print("No data in station.")
            logger.warning(f'Station {self.name} has no data.')
            
            
        else:
            starttimestr = datetime.strftime(min(self.df().index), Settings.print_fmt_datetime)
            endtimestr = datetime.strftime(max(self.df().index), Settings.print_fmt_datetime)
            
        
            print('Observations found for period: ', starttimestr, ' --> ', endtimestr)

        
    def df(self):
        """
        Convert all observations of the station to a pandas dataframe.

        Returns
        -------
        pandas.DataFrame
            A Dataframe containing all observations with a datetime index.

        """
        return pd.DataFrame([self.temp,
                             self.radiation_temp, 
                            self.humidity,
                            self.precip,
                            self.precip_sum,          
                            self.wind_speed,
                            self.wind_gust,
                            self.wind_direction,                   
                            self.pressure,
                            self.pressure_at_sea_level]).transpose()


        
        
    def get_lcz(self):
        logger.debug(f'Extract LCZ for {self.name}.')
        
        geo_templates = Settings.geo_datasets_templates
        lcz_file = Settings.geo_lcz_file
        
        if isinstance(lcz_file, type(None)):
            print('No lcz tif location in the settings. Update settings: ')
            print('settings_obj.update_settings(geotiff_lcz_file="...."')
            logger.error('Extracting LCZ but no geotiff file specified!')
            return
        
        lcz_templates = [geo_templ for geo_templ in geo_templates if geo_templ['usage']=='LCZ']
        
        assert len(lcz_templates)==1, 'More (or no) lcz template found!'
        
        lcz_template = lcz_templates[0]
        
        human_mapper = {num: lcz_template['covers'][num]['cover_name'] 
                        for num in lcz_template['covers'].keys()}
        

        #Check if coordinates are available
        if np.isnan(self.lat.iloc[0]) | np.isnan(self.lon.iloc[0]):
            self.lcz = 'Location unknown'
            logger.error(f'Extracting LCZ but coordiates ({self.lat.iloc[0]}, {self.lon.iloc[0]}) of Station {self.name} not known!')
            return 'Location unknown'
        
        #TODO: lat and lons are time depending, now use first lat, lon

        lcz = geotiff_point_extraction(lat=self.lat.iloc[0],
                                       lon=self.lon.iloc[0],
                                       geotiff_location=lcz_file,
                                       geotiff_crs=lcz_template['epsg'],
                                       class_to_human_mapper=human_mapper)

        self.lcz = lcz
        return lcz

    def make_plot(self, variable='temp', title=None):
        
        """
        Make a timeseries plot of one attribute.

        Parameters
        ----------
        variable : str, optional
            Name of attribute to plot. Must be one of [temp, radiation_temp, humidity, precip, wind_speed wind_gust, wind_direction, pressure, pressure_at_sea_level].
            The default is 'temp'.
        **kwargs : 
            named-arguments that are passed to matplolib.pyplot.plot()

        Returns
        -------
        ax : AxesSubplot
            AxesSubplot is returned so layer can be added to it.

        """
        logger.info(f'Make {variable} plot for Station {self.name}.')
        #Load default plot settings
        default_settings=Settings.plot_settings['time_series']
      
        
        #Make title
        if isinstance(title, type(None)):
            title = self.name + ': ' + self.obs_description[variable]
    
        
        #make figure
        ax = timeseries_plot(dtseries=getattr(self, variable),
                             title=title,
                             xlabel='',
                             ylabel=self.units[variable],
                             figsize=default_settings['figsize'],
                             )
                        
        
        
        return ax
    
    def drop_duplicate_timestamp(self):
        logger.debug(f'Drop duplicate timestamps for {self.name}')
        df = pd.DataFrame()
        for obstype in observation_types:
            df[obstype] = getattr(self, obstype)
            
        
        #check if all datetimes are unique
        if df.index.is_unique:
            return
        
        else:
            logger.warning(f'Duplicate timestamps found for station {self.name}. These first occurances will be kept.')
            print("DUPLICATE TIMESTAMPS FOUND FOR ",self.name)
            df = df.reset_index()
            df = df.rename(columns={'index': 'datetime'})
            df = df.drop_duplicates(subset='datetime')
            df = df.set_index('datetime', drop=True)
        
            #update attributes
            for obstype in observation_types:
                setattr(self, obstype, df[obstype])
                
    def apply_gross_value_check(self, obstype='temp', ignore_val=np.nan):
        
        logger.info(f'Apply gross value check on {obstype} of {self.name}.')
        updated_obs, qc_flags = gross_value(input_series=getattr(self, obstype),
                                                  obstype=obstype,
                                                  ignore_val=ignore_val)
        
        #update obs attributes
        setattr(self, obstype, updated_obs)
        #update qc flags df
        self.qc_labels_df[obstype]['gross_value'] = qc_flags
        
    def apply_persistance_check(self, obstype='temp', ignore_val=np.nan):
        logger.info(f'Apply persistance check on {obstype} of {self.name}.')
        updated_obs, qc_flags = persistance(input_series=getattr(self, obstype),
                                                  obstype=obstype,
                                                  ignore_val=ignore_val)
        
        #update obs attributes
        setattr(self, obstype, updated_obs)
        #update qc flags df
        self.qc_labels_df[obstype]['persistance'] = qc_flags

# =============================================================================
# Dataset class
# =============================================================================

class Dataset:
    def __init__(self):
        logger.info('Initialise dataset')
        self._stationlist = []
        self.df = pd.DataFrame()
        
        self.data_template = {}
        
    
    def get_station(self, stationname):
        
        """
        Extract a station object from the dataset.

        Parameters
        ----------
        stationname : String
            Name of the station, example 'vlinder16'

        Returns
        -------
        station_obj : vlinder_toolkit.Station
            

        """
        logger.info(f'Extract {stationname} from dataset.')
        
        for station_obj in self._stationlist:
            if stationname == station_obj.name:
                return station_obj
            
        logger.warning(f'{stationname} not found in the dataset.')
        print(stationname, ' not found in the dataset!')
    
    def get_geodataframe(self):
        logger.debug('Converting dataset to a geopandas dataframe.')
        gdf = gpd.GeoDataFrame(self.df,
                               geometry=gpd.points_from_xy(self.df['lon'],
                                                           self.df['lat']))
        return gdf
    
    def show(self):
        logger.info('Show basic info of dataset.')
        if self.df.empty:
            print("This dataset is empty!")
            logger.error('The dataset is empty!')
        else: 
            starttimestr = datetime.strftime(min(self.df.index), Settings.print_fmt_datetime)
            endtimestr = datetime.strftime(max(self.df.index), Settings.print_fmt_datetime)
            
            stations_available = list(self.df.name.unique())
        
            print(f'Observations found for period: {starttimestr} --> {endtimestr}')
            logger.debug(f'Observations found for period: {starttimestr} --> {endtimestr}')
            print(f'Following stations are in dataset: {stations_available}')
            logger.debug(f'Following stations are in dataset: {stations_available}')
        
    def make_plot(self, stationnames=None, variable='temp',
                                   starttime=None, endtime=None,
                                   title=None, legend=True):
        """
        This function create a timeseries plot for the dataset. The variable observation type
        is plotted for all stationnames from a starttime to an endtime.

        Parameters
        ----------
        stationnames : List, Iterable
            Iterable of stationnames to plot.
        variable : String, optional
            The name of the observation type to plot. The default is 'temp'.
        starttime : datetime, optional
            The starttime of the timeseries to plot. The default is None and all observations 
            are used.
        endtime : datetime, optional
            The endtime of the timeseries to plot. The default is None and all observations 
            are used..
        title : String, optional
            Title of the figure, if None a default title is generated. The default is None.
        legend : Bool, optional
           Add legend to the figure. The default is True.
        Returns
        -------
        ax : matplotlib.axes
            The plot axes is returned.

        """
        logger.info(f'Make {variable}-timeseries plot for {stationnames}')
        
        default_settings=Settings.plot_settings['time_series']
        
        if isinstance(stationnames, type(None)):
            plotdf = self.df
        else:
            plotdf = self.df[self.df['name'].isin(stationnames)]
        
        #Time subsetting
        plotdf = datetime_subsetting(plotdf, starttime, endtime)
        
        
        relevant_columns = ['name']
        relevant_columns.append(variable)
        plotdf = plotdf[relevant_columns]
        
        plotdf = pd.pivot(plotdf,
                          columns='name',
                          values=variable)
        
        
        if isinstance(title, type(None)):
            title=Settings.display_name_mapper[variable] + ' for stations: ' + str(stationnames)
        
        ax = timeseries_comp_plot(plotdf=plotdf,
                                  title=title,
                                  xlabel='',
                                  ylabel=Settings.display_name_mapper[variable],
                                  figsize=default_settings['figsize'])
        
        return ax
        
        
        
        
        
    def make_geo_plot(self, variable='temp', title=None, timeinstance=None, legend=True,
                      vmin=None, vmax=None):
        """
        This functions creates a geospatial plot for a field (observations or attributes) of all stations.
        
        If the field is timedepending, than the timeinstance is used to plot the field status at that datetime.
        If the field is categorical than the leged will have categorical values, else a colorbar is used. 
        
        All styling attributes are extracted from the Settings.
        

        Parameters
        ----------
        variable : String, optional
            Fieldname to visualise. This can be an observation or station attribute. The default is 'temp'.
        title : String, optional
            Title of the figure, if None a default title is generated. The default is None.
        timeinstance : datetime, optional
            Datetime moment of the geospatial plot. The default is None and the first datetime available
            is used.
        legend : Bool, optional
            Add legend to the figure. The default is True.
        vmin : float, optional
            The minimum value corresponding to the minimum color. The default is None and 
            the minimum of the variable is used.
        vmax : float, optional
           The maximum value corresponding to the minimum color. The default is None and 
           the maximum of the variable is used.

        Returns
        -------
        ax : Geoaxes
            The geoaxes is returned.

        """
        
        
        
        #Load default plot settings
        default_settings=Settings.plot_settings['spatial_geo']
        
        #get first timeinstance of the dataset if not given
        if isinstance(timeinstance, type(None)):
            timeinstance=self.df.index.min()
            
        logger.info(f'Make {variable}-geo plot at {timeinstance}')
        
        #subset to timeinstance
        subdf = self.df.loc[timeinstance]
        
        #check if coordinates are defined
        if (all(subdf['lat'].isnull()) | (all(subdf['lon'].isnull()))):
            print('No coordinates available, add metadata file!')
            return
        
        #create geodf
        gdf = gpd.GeoDataFrame(subdf,
                               geometry=gpd.points_from_xy(subdf['lon'], subdf['lat']))
        
        gdf = gdf[[variable, 'geometry']]
        
    
        

        #make color scheme for field
        if variable in categorical_fields:
            is_categorical=True
            if variable == 'lcz':
                #use all available LCZ categories
                use_quantiles=False
            else:
                use_quantiles=True
        else:
            is_categorical=False
            use_quantiles=False
     
        
        #if observations extend is contained by default exten, use default else use obs extend
        use_extent=find_largest_extent(geodf=gdf,
                                       extentlist=default_settings['extent'])
        
        
        #Style attributes
        if isinstance(title, type(None)):
            if variable in static_fields:
                title = Settings.display_name_mapper[variable]
            else:
                dtstring = datetime.strftime(timeinstance, default_settings['fmt'])
                title = Settings.display_name_mapper[variable] + ' at ' + dtstring
        
        ax = spatial_plot(gdf=gdf,
                          variable=variable,
                          legend=legend,
                          use_quantiles=use_quantiles,
                          is_categorical=is_categorical,
                          k_quantiles=default_settings['n_for_categorical'],
                          cmap = default_settings['cmap'],
                          world_boundaries_map=Settings.world_boundary_map,
                          figsize=default_settings['figsize'],
                          extent=use_extent,
                          title=title,
                          vmin=vmin,
                          vmax=vmax
                          )
        

        return ax
    
    
    def write_to_csv(self, filename=None):
        logger.info('Writing the dataset to a csv file')
        assert not isinstance(Settings.output_folder, type(None)), 'Specify Settings.output_folder in order to export a csv.'
        
        #update observations with QC labels etc        
        self._update_dataset_df_with_stations()
        
        #Get observations and metadata columns in the right order
        logger.debug('Merging data and metadata')
        df_columns = observation_types.copy()
        df_columns.extend(location_info)
        writedf = self.df[df_columns]
        
                
        #find observation type that are not present
        ignore_obstypes = [col for col in observation_types if writedf[col].isnull().all()]
        
        writedf = writedf.drop(columns=ignore_obstypes)
        
        logger.debug(f'Skip quality labels for obstypes: {ignore_obstypes}.')
        
        writedf.index.name = 'datetime'
        
        #add final quality labels
        
        final_labels = pd.DataFrame()
        for station in self._stationlist:
            logger.debug(f'Computing a final quality label for {station.name} observations')
            station_final_labels = pd.DataFrame()
            for obstype in observation_types:
                if obstype in ignore_obstypes:
                    continue
                station_final_labels[obstype + '_QC_label'] = final_qc_label_maker(station.qc_labels_df[obstype],
                                                      Settings.qc_numeric_label_mapper)
              
            station_final_labels['name'] = station.name
            final_labels = pd.concat([final_labels, station_final_labels])    
        final_labels.index.name = 'datetime'
        writedf = writedf.merge(right=final_labels,
                                    how='left',
                                    on=['datetime', 'name'])
        
        #make filename
        if isinstance(filename, type(None)):
            startstr = self.df.index.min().strftime('%Y%m%d') 
            endstr = self.df.index.max().strftime('%Y%m%d') 
            filename= 'dataset_' + startstr + '_' + endstr
        else:
            if filename.endswith('.csv'):
                filename = filename[:-4] #to avoid two times .csv.csv
            
        filepath = os.path.join(Settings.output_folder, filename + '.csv')
        
        #write to csv in output folder
        logger.info(f'write dataset to file: {filepath}')
        writedf.to_csv(path_or_buf=filepath,
                       sep=';',
                       na_rep='NaN',
                       index=True)        
        
    
    # =============================================================================
    # Update dataset by station objects
    # =============================================================================
    
    def _update_dataset_df_with_stations(self):
        logger.debug('Updating dataset.df from stationlist')
    
        present_df_columns = list(self.df.columns)    
        updatedf = pd.DataFrame()
        for station in self._stationlist:
            stationdf = station.df() #start with observations
            
            #add meta data
            for attr in present_df_columns:
                if attr in stationdf.columns:
                    continue #skip observations because they are already in the df
                try:
                    stationdf[attr] = getattr(station,attr)
                except:
                    stationdf[attr] = 'not updated'
            
            updatedf = pd.concat([updatedf, stationdf])
        
        
        updatedf = updatedf[present_df_columns] #reorder columns
        self.df = updatedf
        return
                
            
            


    
    # =============================================================================
    #     Quality control
    # =============================================================================
    
    def apply_quality_control(self, obstype='temp',
                              gross_value=True, persistance=True, ignore_val=np.nan):
        """
        Apply quality control methods to the dataset. The default settings are used, and can be changed
        in the settings_files/qc_settings.py
        
        The checks are performed in a sequence: gross_vallue --> persistance --> ...,
        Outliers by a previous check are ignored in the following checks!
        
        The dataset and all it stations are updated inline.

        Parameters
        ----------
        obstype : String, optional
            Name of the observationtype you want to apply the checks on. The default is 'temp'.
        gross_value : Bool, optional
            If True the gross_value check is applied if False not. The default is True.
        persistance : Bool, optional
           If True the persistance check is applied if False not. The default is True.. The default is True.
        ignore_val : numeric, optional
            Values to ignore in the quality checks. The default is np.nan.

        Returns
        -------
        None.

        """
        
        
        if gross_value:
            print('Applying the gross value check on all stations.')
            logger.info('Applying gross value check on the full dataset')
            for stationobj in self._stationlist:
                stationobj.apply_gross_value_check(obstype=obstype,
                                                   ignore_val=ignore_val)
                
        if persistance:
            print('Applying the persistance check on all stations.')
            logger.info('Applying persistance check on the full dataset')
            for stationobj in self._stationlist:
                stationobj.apply_persistance_check(obstype=obstype,
                                                   ignore_val=ignore_val)

        #update the dataframe with stations values
        self._update_dataset_df_with_stations()


    # =============================================================================
    #     importing data        
    # =============================================================================
            
    def import_data_from_file(self, network='vlinder', coarsen_timeres=False):
        """
        Read observations from a csv file as defined in the Settings.input_file. 
        The network and stations objects are updated. It is possible to apply a 
        resampling (downsampling) of the observations as defined in the settings.

        Parameters
        ----------
        network : String, optional
            The name of the network for these observationsThe default is 'vlinder'.
        coarsen_timeres : Bool, optional
            If True, the observations will be interpolated to a coarser time resolution
            as is defined in the Settings. The default is False.

        Returns
        -------
        None.

        """
        print('Settings input data file: ', Settings.input_data_file)
        logger.info(f'Importing data from file: {Settings.input_data_file}')
        
        # Read observations into pandas dataframe
        df, template = import_data_from_csv(input_file = Settings.input_data_file,
                                  file_csv_template=Settings.input_csv_template,
                                  template_list = Settings.template_list)

        logger.debug(f'Data from {Settings.input_data_file} imported to dataframe.')

        #drop Nat datetimes if present
        df = df.loc[pd.notnull(df.index)]
        
        
        if isinstance(Settings.input_metadata_file, type(None)):
            print('WARNING: No metadata file is defined. Add your settings object.')
            logger.warning('No metadata file is defined, no metadata attributes can be set!')
        else:
            logger.info(f'Importing metadata from file: {Settings.input_metadata_file}')
            meta_df = import_metadata_from_csv(input_file=Settings.input_metadata_file,
                                               file_csv_template=Settings.input_metadata_template,
                                               template_list = Settings.template_list)
            
            #merge additional metadata to observations
            meta_cols = [colname for colname in meta_df.columns if not colname.startswith('_')]
            additional_meta_cols = list(set(meta_cols).difference(df.columns))
            if bool(additional_meta_cols):
                logger.debug(f'Merging metadata ({additional_meta_cols}) to dataset data by name.')
                additional_meta_cols.append('name') #merging on name
                df_index = df.index #merge deletes datetime index somehow? so add it back on the merged df
                df = df.merge(right=meta_df[additional_meta_cols],
                              how='left', 
                              on='name')
                df.index = df_index
        
        
        #update dataset object
        self.data_template = template
        
        
        
        if coarsen_timeres:
            logger.info(f'Coarsen timeresolution to {Settings.target_time_res} using the {Settings.resample_method}-method.')
            df = coarsen_time_resolution(df=df,
                                          freq=Settings.target_time_res,
                                          method=Settings.resample_method)
            
        
        self.update_dataset_by_df(df)
        
    
    def import_data_from_database(self,
                              start_datetime=None,
                              end_datetime=None,
                              coarsen_timeres=False):
        """
        Function to import data directly from the framboos database and updating the 
        network and station objects. 
        

        Parameters
        ----------
        start_datetime : datetime, optional
            Start datetime of the observations. The default is None and using 
            yesterday's midnight.
        end_datetime : datetime, optional
            End datetime of the observations. The default is None and using todays
            midnight.
        coarsen_timeres : Bool, optional
            If True, the observations will be interpolated to a coarser time resolution
            as is defined in the Settings. The default is False.

        Returns
        -------
        None.

        """
        if isinstance(start_datetime, type(None)):
            start_datetime=datetime.date.today() - datetime.timedelta(days=1)
        if isinstance(end_datetime, type(None)):
            end_datetime=datetime.date.today()
       
            
        # Read observations into pandas dataframe
        df = import_data_from_database(Settings,
                                       start_datetime=start_datetime,
                                       end_datetime=end_datetime)
        
        
        #Make data template
        self.data_template = template_to_package_space(Settings.vlinder_db_obs_template)
        
        if coarsen_timeres:
            df = coarsen_time_resolution(df=df,
                                          freq=Settings.target_time_res,
                                          method=Settings.resample_method)
        
        self.update_dataset_by_df(df)
    
    
    
            
    def update_dataset_by_df(self, dataframe):
        """
        Update the dataset object and creation of station objects and all it attributes by a dataframe.
        This is done by initialising stations and filling them with observations and meta
        data if available. 
        
        When filling the observations, there is an automatic check for missing timestamps. 
        If a missing timestamp is detected, the timestamp is created with Nan values for all observation types.
        

        Parameters
        ----------
        dataframe : pandas.DataFrame
        A dataframe that has an datetimeindex and following columns: 'name, temp, radiation_temp, humidity, ...'
            

        Returns
        -------
        None.

        """
       
        logger.info(f'Updating dataset by dataframe with shape: {dataframe.shape}.')
        #reset dataset attributes
        
        
        present_observation_types = [obstype for obstype in dataframe.columns if obstype in observation_types]
       
        #make shure all columns are present (nan's if no data) and always the same structure
        df_columns = observation_types.copy()
        df_columns.extend(location_info)
        
        # Needed for sorting of columns
        missing_columns = list(set(df_columns).difference(set(dataframe.columns)))
        for missing_col in missing_columns:
            dataframe[missing_col] = np.nan

        self.df = dataframe
        self._stationlist = [] 
        
      
        
        
        
        # Create a list of station objects
        for stationname in dataframe.name.unique():
            logger.debug(f'Extract data of {stationname}.')
            #extract observations
            station_obs = dataframe[dataframe['name'] == stationname].sort_index()
            
            if station_obs.empty:
                logger.warning(f'{stationname} has no data. This station will be ignored.')
                print('skip stationname: ', stationname)
                continue
            
            #find network
            if 'network' in station_obs.columns:
                network = station_obs['network'].iloc[0]
            else:
                if 'linder' in stationname:
                    network='vlinder'
                elif 'occa' in stationname:
                    network='mocca'
                else:
                    network='Unknown'
            logger.debug(f'Network of {stationname} is {network}.')
            
            
            
            #initialise station object
            station_obj = Station(station_name=stationname, 
                                  network_name=network)
            
            logger.debug(f'fill data attributes for {stationname}')
            #add observations to the attributes
            for obstype in observation_types:
                #fill attributes of station object
                try:
                    setattr(station_obj, obstype, station_obs[obstype])
                except KeyError:
                    # example: in the mocca network there is no column of radiation temp
                    continue
                
            #drop duplicate timestamps    
            station_obj.drop_duplicate_timestamp()
            
            
            

            #Apply IO checks
            
            #check for missing timestamps
            logger.debug(f'Check for missing timestamps in the inputfile for {stationname}.')
            checked_df, statusdf = missing_timestamp_check(station_obj)
            for obstype in checked_df.columns:
                #update observations with missing obs as nan's
                try:
                    setattr(station_obj, obstype, checked_df[obstype])
                    #update QC dataframes
                    station_obj.qc_labels_df[obstype] = pd.DataFrame(data = {'observations': checked_df[obstype]})
                    #the next line is to slow!
                    station_obj.qc_labels_df[obstype]['status'] = statusdf['status']
                except KeyError:
                    continue
            
            logger.debug(f'Create no-observations labels df for {stationname}.')
            for obstype in observation_types:
                try:
                    if not (obstype in present_observation_types):
                        station_obj.qc_labels_df[obstype]['status'] = 'no observations'
                except KeyError:
                    continue
            
            
     
            logger.debug(f'add metadata to {stationname} station.')
            #check if meta data is available
            if 'lat' in station_obs.columns:
                station_obj.lat = station_obs['lat']
                check_for_nan(station_obj.lat, 'latitude', stationname)
            if 'lon' in station_obs.columns:
                station_obj.lon = station_obs['lon']
                check_for_nan(station_obj.lon, 'longtitude', stationname)
            if 'call_name' in station_obs.columns:
                station_obj.call_name = station_obs['call_name'].iloc[0]
                check_for_nan(station_obj.call_name, 'call_name', stationname)
            if 'location' in station_obs.columns:
                station_obj.location = station_obs['location'].iloc[0]
                check_for_nan(station_obj.location, 'location', stationname)
                
            # Get physiography data if possible
            if not isinstance(Settings.geo_lcz_file, type(None)):
                try:
                    _ = station_obj.get_lcz()
                    logger.debug(f'Extracting LCZ for {stationname}: {_}.')
                except:
                    _=None
                    logger.warning(f' LCZ could not be extracted for {stationname}.')
            
            #Update units and description dicts of the station using the used template
            for obs_field in station_obj.units.keys():
                try:
                    station_obj.units[obs_field] = self.data_template[obs_field]['units']
                    station_obj.obs_description[obs_field] = self.data_template[obs_field]['description']
                except KeyError:
                   continue 
            
            
            #update stationlist
            self._stationlist.append(station_obj)
            
        
        #Update dataset df with information created on station level
        
        #add LCZ to dataset df
        logger.debug('Add LCZ info to the dataset dataframe.')
        if not isinstance(Settings.geo_lcz_file, type(None)): 
            lcz_dict = {station.name: station.lcz for station in self._stationlist}
            self.df['lcz'] = self.df['name'].map(lcz_dict)
        
            
          
def check_for_nan(value, fieldname, stationname):
    """
    Check for nan values in a input value that has a fieldname. Nothing is done to 
    the input value, only print statements
    Parameters
    ----------
    value : float or pd.Series
        value(s) to test.
    fieldname : string
        the name of the variable    
    stationname : string
        name of the station
    Returns
    -------
    None.

    """
    if isinstance(value, float):
        if np.isnan(value):
            print('Nan found for ', fieldname, ' in ', stationname, '!!')
            logger.warning(f'Missing {fieldname} for {stationname}: {value}.')
    elif isinstance(value, pd.Series):
        if value.isnull().sum() > 0:
            n_nans = value.isnull().sum()
            print(n_nans, "Nan's found in ", fieldname, '-iterable in ', stationname, '!!')
            logger.warning(f'{n_nans} Missing {fieldname} foud for {stationname}.')
        

        
def missing_timestamp_check(station):
    """
    Looking for missing timestaps by assuming an observation frequency. The assumed frequency is the most occuring frequency.
    If missing observations are detected, the observations dataframe is extended by these missing timestamps with Nan's as filling values.

    Parameters
    ----------
    station : Station object
        The station you whant to apply this check on.

    Returns
    -------
    df : pandas.DataFrame()
        The observations dataframe (same as Station.df()).
    missing_datetimes : list of datetimes
        The list of the missing timestamps.

    """     
   
    
    df = station.df()
   
    #extrac observed frequencies
    likely_freq = df.index.to_series().diff().value_counts().idxmax()
    
    
    missing_datetimeindices = pd.date_range(start = df.index.min(),
                         end = df.index.max(),
                         freq=likely_freq).difference(df.index)
    
    if not missing_datetimeindices.empty:
        logging.warning(f'{len(missing_datetimeindices)} missing records ({missing_datetimeindices[:10]} ...) found for {station.name}. These will be filled with Nans.')
    
    
    
    statusdf = pd.concat([pd.DataFrame(data='ok', index=df.index, columns=['status']),
                          pd.DataFrame(data='missing timestamp', index=missing_datetimeindices, columns=['status'])])
    
    missing_df = pd.DataFrame(data=np.nan,
                              index=missing_datetimeindices,
                              columns=df.columns)
    
    df = pd.concat([df, missing_df])
    
    
    df = df.sort_index()
    
    
    return df, statusdf
    

def datetime_subsetting(df, starttime, endtime):
    """
    Wrapper function for subsetting a dataframe with datetimeindex with a start- and 
    endtime. 

    Parameters
    ----------
    df : pandas.DataFrame with datetimeindex
        The dataframe to apply the subsetting to.
    starttime : datetime.Datetime
        Starttime for the subsetting period (included).
    endtime : datetime.Datetime
        Endtime for the subsetting period (included).

    Returns
    -------
    pandas.DataFrame
        Subset of the df.

    """
    
    stand_format = '%Y-%m-%d %H:%M:%S'
    
    if isinstance(starttime, type(None)):
        startstring = None #will select from the beginning of the df
    else:
        startstring = starttime.strftime(stand_format)
    if isinstance(endtime, type(None)):
        endstring = None
    else: 
        endstring = endtime.strftime(stand_format)

    return df[startstring: endstring]

def final_qc_label_maker(qc_df, label_to_numeric_mapper):
    """
    This function creates a final label based on de individual qc labels. If all labels
    are ok, the final label is ok. Else the final label will be that of the individual qc-label
    which rejected the obseration.

    Parameters
    ----------
    qc_df : pandas.DataFrame 
        the specific qc_label_df with the datetimeindex, the first column the observations,
        and labels for each QC check per column.
    label_to_numeric_mapper : dict
        The dictionary that maps qc-labels to numeric values (for speedup).

    Returns
    -------
    final_labels : pd.Series
        A series with the final labels and the same index as the qc_df.

    """ 
    qc_labels_columns = qc_df.columns.to_list()[1:] #ignore the observations
    
    #map columns to numeric
    num_label_df = pd.DataFrame()
    for qc_column in qc_labels_columns:
        num_label_df[qc_column] = qc_df[qc_column].map(label_to_numeric_mapper)

    #invert numeric mapper
    inv_label_to_num = {v: k for k, v in label_to_numeric_mapper.items()}
    
    final_labels = num_label_df.sum(axis=1, skipna=True).map(inv_label_to_num)
    return final_labels