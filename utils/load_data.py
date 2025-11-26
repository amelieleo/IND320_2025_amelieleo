from __future__ import annotations

import pandas as pd
import streamlit as st
import pymongo
from retry_requests import retry
import requests_cache
import openmeteo_requests

#--------------------------------------------------------------------------------------
#---------------------------WEATHER DATA-----------------------------------------------
#--------------------------------------------------------------------------------------

@st.cache_data(show_spinner=True)
def load_weather_data(price_area: str = "NO1", year: int = 2021, latitude: float | None = None, longitude: float | None = None) -> pd.DataFrame:
    with st.spinner("Loading weather data..."):
        cities = [
            {"Price Area Code": "NO1", "Representative City": "Oslo", "Latitude": 59.911491, "Longitude": 10.757933},
            {"Price Area Code": "NO2", "Representative City": "Kristiansand", "Latitude": 58.1467, "Longitude": 7.9956},
            {"Price Area Code": "NO3", "Representative City": "Trondheim", "Latitude": 63.42, "Longitude": 10.45},
            {"Price Area Code": "NO4", "Representative City": "Tromsø", "Latitude": 69.649208, "Longitude": 18.954343},
            {"Price Area Code": "NO5", "Representative City": "Bergen", "Latitude": 60.392078, "Longitude": 5.327885}
        ]

        # Create the pandas DataFrame
        price_areas = pd.DataFrame(cities)

        #city as index
        price_areas.set_index('Price Area Code', inplace=True)

        if latitude is None or longitude is None:
            lat = price_areas.loc[price_area, "Latitude"]
            lon = price_areas.loc[price_area, "Longitude"]
        else:
            lat, lon = latitude, longitude
            
        cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
        retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
        openmeteo = openmeteo_requests.Client(session = retry_session)

        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ["precipitation", "temperature_2m", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"],
            "models": "era5",
            "timezone": "Europe/Oslo",
        }
        responses = openmeteo.weather_api(url, params=params)

        response = responses[0]

        # Process hourly data. The order of variables needs to be the same as requested.
        hourly = response.Hourly()
        dt_index_utc = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=int(hourly.Interval())),
            inclusive="left",
        )
        dt_index_oslo = dt_index_utc.tz_convert("Europe/Oslo")  # +01 in winter, +02 in summer

        hourly_data = {
            "date": dt_index_oslo,
            "precipitation": hourly.Variables(0).ValuesAsNumpy(),
            "temperature_2m": hourly.Variables(1).ValuesAsNumpy(),
            "wind_speed_10m": hourly.Variables(2).ValuesAsNumpy(),
            "wind_direction_10m": hourly.Variables(3).ValuesAsNumpy(),
            "wind_gusts_10m": hourly.Variables(4).ValuesAsNumpy(),
        }
        hourly_dataframe = pd.DataFrame(hourly_data)
        hourly_dataframe.set_index("date", inplace=True)
        hourly_dataframe.sort_index(inplace=True)
        st.info(
            f"Loaded weather data for {price_areas.loc[price_area]['Representative City']} "
            f"({price_area}) • lat {lat:.3f}, lon {lon:.3f}"
        )
    
    return hourly_dataframe


#------------------------------------------------------------------------------------
# ---------------------------ENERGY PRODUCTION DATA----------------------------------
# -----------------------------------------------------------------------------------   


#connecting to MongoDB
@st.cache_resource #Connecting to MongoDB
def init_connection():
    return pymongo.MongoClient(st.secrets["mongo"]["uri"])


#Loading the data from the MongoDB database
@st.cache_data(ttl=600, show_spinner=False)
def load_energy_production_data(col_name: int = 2021) -> pd.DataFrame:
    with st.spinner("Loading the data..."):
        client = init_connection()
        db = client['electricity_production']
        collection = db[f"data_{col_name}"]
        items = collection.find()
        items = pd.DataFrame(list(items))
    return items

@st.cache_data(ttl=600, show_spinner=False)
def load_energy_consumption_data(col_name: int = 2021) -> pd.DataFrame:
    with st.spinner("Loading the data..."):
        client = init_connection()
        db = client['electricity_consumption']
        collection = db[col_name]
        items = collection.find()
        items = pd.DataFrame(list(items))
    return items
