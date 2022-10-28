#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The class object for a Vlinder/mocca station
@author: thoverga
"""
from collections.abc import Iterable

import pandas as pd
import geopandas as gpd
import numpy as np

from datetime import datetime

from .settings import Settings
from .data_import import import_data_from_csv, import_data_from_database, template_to_package_space, import_metadata_from_csv
from .data_import import coarsen_time_resolution
from .landcover_functions import geotiff_point_extraction
from .geometry_functions import find_largest_extent
from .plotting_functions import spatial_plot, timeseries_plot, timeseries_comp_plot
from .qc_checks import gross_value, persistance


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


# =============================================================================
# station class
# =============================================================================
class Station:
    def __init__(self, station_name, network_name):
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
        
        geo_templates = Settings.geo_datasets_templates
        lcz_file = Settings.geo_lcz_file
        
        if isinstance(lcz_file, type(None)):
            print('No lcz tif location in the settings. Update settings: ')
            print('settings_obj.update_settings(geotiff_lcz_file="...."')
            return
        
        lcz_templates = [geo_templ for geo_templ in geo_templates if geo_templ['usage']=='LCZ']
        
        assert len(lcz_templates)==1, 'More (or no) lcz template found!'
        
        lcz_template = lcz_templates[0]
        
        human_mapper = {num: lcz_template['covers'][num]['cover_name'] 
                        for num in lcz_template['covers'].keys()}
        

        #Check if coordinates are available
        if np.isnan(self.lat.iloc[0]) | np.isnan(self.lon.iloc[0]):
            self.lcz = 'Location unknown'
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
        
        df = pd.DataFrame()
        for obstype in observation_types:
            df[obstype] = getattr(self, obstype)
            
        
        #check if all datetimes are unique
        if df.index.is_unique:
            return
        
        else:
            print("DUPLICATE TIMESTAMPS FOUND FOR ",self.name)
            df = df.reset_index()
            df = df.rename(columns={'index': 'datetime'})
            df = df.drop_duplicates(subset='datetime')
            df = df.set_index('datetime', drop=True)
        
            #update attributes
            for obstype in observation_types:
                setattr(self, obstype, df[obstype])
                
    def apply_gross_value_check(self, obstype='temp', ignore_val=np.nan):
        updated_obs, qc_flags = gross_value(input_series=getattr(self, obstype),
                                                  obstype=obstype,
                                                  ignore_val=ignore_val)
        
        #update obs attributes
        setattr(self, obstype, updated_obs)
        #update qc flags df
        self.qc_labels_df[obstype]['gross_value'] = qc_flags
        
    def apply_persistance_check(self, obstype='temp', ignore_val=np.nan):
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
        for station_obj in self._stationlist:
            if stationname == station_obj.name:
                return station_obj
        
        print(stationname, ' not found in the dataset!')
    
    def get_geodataframe(self):
        gdf = gpd.GeoDataFrame(self.df,
                               geometry=gpd.points_from_xy(self.df['lon'],
                                                           self.df['lat']))
        return gdf
    
    def show(self):
        if self.df.empty:
            print("This dataset is empty!")
        else: 
            starttimestr = datetime.strftime(min(self.df.index), Settings.print_fmt_datetime)
            endtimestr = datetime.strftime(max(self.df.index), Settings.print_fmt_datetime)
            
            stations_available = list(self.df.name.unique())
        
            print('Observations found for period: ', starttimestr, ' --> ', endtimestr)
            print('Following stations are in dataset: ', stations_available)
        
        
    def make_plot(self, stationnames, variable='temp',
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
        
        default_settings=Settings.plot_settings['time_series']
        
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
        
        #subset to timeinstance
        subdf = self.df.loc[timeinstance]
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
    
    # =============================================================================
    # Update dataset by station objects
    # =============================================================================
    
    def _update_dataset_df_with_stations(self):
    
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
            for stationobj in self._stationlist:
                stationobj.apply_gross_value_check(obstype=obstype,
                                                   ignore_val=ignore_val)
                
        if persistance:
            print('Applying the persistance check on all stations.')
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
        
        
        # Read observations into pandas dataframe
        df, template = import_data_from_csv(input_file = Settings.input_data_file,
                                  file_csv_template=Settings.input_csv_template,
                                  template_list = Settings.template_list)
        #drop Nat datetimes if present
        df = df.loc[pd.notnull(df.index)]
        
        if isinstance(Settings.input_metadata_file, type(None)):
            print('WARNING: No metadata file is defined. Add your settings object.')
        else:
            meta_df = import_metadata_from_csv(input_file=Settings.input_metadata_file,
                                               file_csv_template=Settings.input_metadata_template,
                                               template_list = Settings.template_list)
            
            #merge additional metadata to observations
            meta_cols = [colname for colname in meta_df.columns if not colname.startswith('_')]
            additional_meta_cols = list(set(meta_cols).difference(df.columns))
            if bool(additional_meta_cols):
                additional_meta_cols.append('name') #merging on name
                df_index = df.index #merge deletes datetime index somehow? so add it back on the merged df
                df = df.merge(right=meta_df[additional_meta_cols],
                              how='left', 
                              on='name')
                df.index = df_index
        
        
        #update dataset object
        self.data_template = template
        
        
        
        if coarsen_timeres:
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
       
        
        #reset dataset attributes
        self.df = dataframe
        self._stationlist = [] 
        
      
        
        
        
        # Create a list of station objects
        for stationname in dataframe.name.unique():
            #extract observations
            station_obs = dataframe[dataframe['name'] == stationname].sort_index()
            
            if station_obs.empty:
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
            
            
            
            #initialise station object
            station_obj = Station(station_name=stationname, 
                                  network_name=network)
            
            
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
            
         
            for obstype in observation_types:
                try:
                    #Fill QC dataframes with observations
                    station_obj.qc_labels_df[obstype] = pd.DataFrame(data = {'observations': station_obs[obstype]})
                    station_obj.qc_labels_df[obstype]['status'] = 'ok'
                except KeyError:
                    continue

            #Apply IO checks
            
            #check for missing timestamps
            checked_df, missing_dt_list = missing_timestamp_check(station_obj)
            if bool(missing_dt_list):
                for obstype in checked_df.columns:
                    #update observations with missing obs as nan's
                    try:
                        setattr(station_obj, obstype, checked_df[obstype]) 
                        #update QC dataframes
                        station_obj.qc_labels_df[obstype] = pd.DataFrame(data = {'observations': checked_df[obstype]})
                        station_obj.qc_labels_df[obstype]['status'] = ['ok' if dt not in missing_dt_list else 'missing timestamp' for dt in checked_df.index ]
                    except KeyError:
                        continue
                    
                    
                    
            
            
            
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
                except:
                    _=None
            
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
    elif isinstance(value, pd.Series):
        if value.isnull().sum() > 0:
            n_nans = value.isnull().sum()
            print(n_nans, "Nan's found in ", fieldname, '-iterable in ', stationname, '!!')
        

        
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
    
    missing_df = pd.DataFrame(data=np.nan,
                              index=missing_datetimeindices,
                              columns=df.columns)
    
    df = pd.concat([df, missing_df])
    
    
    df = df.sort_index()
    
    
    return df, missing_datetimeindices.to_list()
    

def datetime_subsetting(df, starttime, endtime):
    
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
    
