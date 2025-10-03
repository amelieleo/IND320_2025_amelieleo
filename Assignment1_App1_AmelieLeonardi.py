import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from windrose import WindroseAxes
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.lines as mlines


#------------------------------------------------------------------------------------------
#----------------------------THE DATA------------------------------------------------------
#------------------------------------------------------------------------------------------
@st.cache_data
def load_data():
    data_file = 'data/open-meteo-subset.csv'
    dataframe = pd.read_csv(data_file, index_col='time')
    dataframe.index = pd.to_datetime(dataframe.index)
    return dataframe

data = load_data()

#-----------------------------------------------------------------------------------------
#----------------------------PLOTTING FUNCTIONS-------------------------------------------
#-----------------------------------------------------------------------------------------

#Plot the Temperature with freezing point
def plot_temp(dataframe):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(dataframe.index, dataframe['temperature_2m (°C)'], label='Temperature (°C)', color="#C4611A")
    #plotting a horizontal line at 0°C
    ax.hlines(y=0, xmin=dataframe.index.min(), xmax=dataframe.index.max() + pd.Timedelta(weeks=1), colors="#542F2F", linestyles='dashed', label='Freezing Point (0°C)')

    #setting limits for x axis
    ax.set_xlim([dataframe.index.min(), dataframe.index.max() + pd.Timedelta(weeks=1)])

    #Makes the plot nice and readable
    ax.grid()
    ax.legend()
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Temperature over time with freezing point')
    return fig

#Plot the percipitation as daily total as bar chart --------------------------------------------
def plot_percipitation(dataframe):
    # Group by date and sum precipitation
    daily_precip = dataframe['precipitation (mm)'].groupby(dataframe.index.date).sum()

    #plotting the daily percipitation data
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(daily_precip.index, daily_precip.values, color="#3173EE", width=1.2) #bar plot for daily percipitation

    #making the plot nice and readable
    ax.set_xlabel('Date')
    ax.set_ylabel('Total Precipitation (mm)')
    ax.set_title('Daily Total Precipitation (mm)')
    ax.grid(axis='y')
    plt.xticks(rotation=45)
    return fig

#Plot the wind speed ----------------------------------------------------------------------------
def plot_wind_speed(dataframe):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(dataframe.index, dataframe['wind_speed_10m (m/s)'], color="#AD4DE0")

    #setting limits for x and y axis
    ax.set_xlim([dataframe.index.min(), dataframe.index.max() + pd.Timedelta(weeks=1)])
    ax.set_ylim(bottom=0)

    #making the plot nice and readable
    ax.grid()
    ax.set_xlabel('Time')
    ax.set_ylabel('Wind Speed (m/s)')
    ax.set_title('Wind speed in m/s over time')
    return fig

#plot wind gusts --------------------------------------------------------------------------------
def plot_wind_gusts(dataframe):
    #plotting the wind gust data
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(dataframe.index, dataframe['wind_gusts_10m (m/s)'], color="#3C1053")

    #setting limits for x and y axis
    ax.set_xlim([dataframe.index.min(), dataframe.index.max()])
    ax.set_ylim(bottom=0)

    #making the plot nice and readable
    ax.grid()
    ax.set_xlabel('Time')
    ax.set_ylabel('Wind gusts (m/s)')
    ax.set_title('Wind gusts in m/s over time')
    return fig

#plot the wind direction: over time and windrose -------------------------------------------------
def plot_wind_direction(dataframe):
    wind_dir = dataframe['wind_direction_10m (°)']
    wind_spd = dataframe['wind_speed_10m (m/s)']

    # Create a figure with two subplots: time series and windrose
    fig = plt.figure(figsize=(12, 8))

    # Subplot 1: Time series
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.scatter(dataframe.index, wind_dir, color="#477B65")
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Wind Direction (°)')
    ax1.set_title('Wind Direction Over Time')
    ax1.grid(True)

    # Subplot 2: Windrose
    ax2 = WindroseAxes.from_ax(fig=fig, rect=[0.18, 0.08, 0.65, 0.55])
    ax2.bar(
        wind_dir,
        wind_spd,
        normed=True,
        opening=0.8,
        edgecolor="#FFFFFF",
        cmap=plt.get_cmap('viridis')
    )
    ax2.set_title('Windrose - Wind Speed and Wind Direction')
    ax2.set_legend(title="Wind speed (m/s)")

    plt.tight_layout(rect=[0, 0.5, 1, 1])  # Adjust so windrose doesn't overlap
    return fig

# Combined plot with weekly statistics and daily precipitation ------------------------------------------------------
def plot_all(dataframe):

    #making a rule for resampling the data weekly, starting on the same weekday as the first entry in the dataframe
    first_weekday = dataframe.index[0].strftime('%a').upper()[:3]
    rule = f'W-{first_weekday}'

    #resampling the data weekly to get mean, min and max values
    weekly_mean = dataframe.resample(rule).mean()
    weekly_min = dataframe.resample(rule).min()
    weekly_max = dataframe.resample(rule).max()
    #getting daily precipitation again
    daily_precip = dataframe['precipitation (mm)'].groupby(dataframe.index.date).sum()

    #calculating u and v components for wind direction arrows
    weekly_dir_rad = np.deg2rad(weekly_mean['wind_direction_10m (°)'])
    u = np.cos(weekly_dir_rad)
    v = np.sin(weekly_dir_rad)

    #setting up the plot
    cols = ['wind_gusts_10m (m/s)', 'wind_speed_10m (m/s)', 'temperature_2m (°C)']
    colors = ['#3C1053', '#AD4DE0', '#C4611A']

    fig, ax = plt.subplots(figsize=(10, 8))
    #plotting a horizontal line at 0 for oreintation
    ax.hlines(y=0, xmin=dataframe.index.min(), xmax=dataframe.index.max(), colors="#542F2F", linestyles='dashed')

    #plotting the weekly mean, min and max values of wind gusts, wind speed and temperature
    for idx, col in enumerate(cols):
        ax.plot(weekly_mean.index, weekly_mean[col], label=col, color=colors[idx])
        ax.fill_between(
            weekly_mean.index,
            weekly_min[col],
            weekly_max[col],
            alpha=0.2,
            color=colors[idx]
        )
    #plotting daily precipitation as bars in the background
    ax.bar(daily_precip.index, daily_precip.values, color="#3173EE", width=1.2)

    #adding wind direction arrows at the top of the plot
    ax.quiver(
        weekly_mean.index,
        [ax.get_ylim()[1]] * len(u),
        u, v,
        angles='xy', scale_units='xy', scale=0.35, color='black', width=0.0015
    )

    # Custom legend handles
    wind_patch = mlines.Line2D([], [], color='black', marker=r'$\rightarrow$', linestyle='None', markersize=12, label='Wind Direction')
    precip_patch = mpatches.Patch(color="#3173EE", label="Precipitation (mm/day)")

    # Add to legend
    ax.legend(handles=[*ax.get_legend_handles_labels()[0], wind_patch, precip_patch], loc='lower center')

    #making the plot nice and readable
    ax.set_title('Weekly Weather Statistics with Daily Precipitation')
    ax.set_ylabel('Temperature (°C), Wind Speed/Gusts (m/s), Precipitation (mm)')
    ax.grid()
    ax.set_xlim([dataframe.index.min(), dataframe.index.max()])
    ax.set_xlabel('Time')   
    return fig

#----------------------------------------------------------------------------------
#----------------------STREAMLIT APP-----------------------------------------------
#---------------------------------------------------------------------------------

#sidebar menu with navigation options to the other pages of the app
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to", ['Home', 'Data', 'Visualization', 'Fun'])

if options == 'Home': #-----------------HOME PAGE-----------------------------------
    st.title("Welcome to the Weather Data App")
    st.write("This app allows you to explore and visualize weather data.")
    st.image("https://t4.ftcdn.net/jpg/02/40/24/81/360_F_240248152_piluBt47ZD46vprw7C0xQ88Lk4zXLg81.jpg")

elif options == 'Data': #-----------------DATA PAGE---------------------------------
    st.title("Data")
    st.text("Here is the weather data for the first month in the dataset:")
    # Filter the first month
    first_month = data.index[0].month
    first_year = data.index[0].year
    first_month_data = data[(data.index.month == first_month) & (data.index.year == first_year)]
    # Transpose so each variable is a row, each time is a column
    first_month_data = first_month_data.T
    # Add a column with the time series as a list for each variable
    first_month_data['values'] = first_month_data.values.tolist()
    # Reset index so variable names are a column
    first_month_data = first_month_data.reset_index().rename(columns={'index': 'Variable'})
    # Show in data_editor with a row-wise LineChartColumn
    st.data_editor(
        first_month_data[['Variable', 'values']],
        column_config={
            "values": st.column_config.LineChartColumn(
                "Time Series (First Month)",
                width="medium",
                help="Time series for the first month"
            ),
        },
        hide_index=True,
    )

elif options == 'Visualization': #-----------------VISUALIZATION PAGE---------------------------------
    st.title("Visualization")
    
    # Selectbox to choose a variable to visualize
    variable = st.selectbox(
    "What data would you like to visualize?",
    ("temperature", "percipitation", "wind speed", "wind gusts", "wind direction", "All variables"),
    index=None, 
    placeholder="Select an option",
    help="Choose the weather variable you want to see plotted"
    )

    #Slider to choose which months to visualize
    month = st.slider("Select month", 1, 12, (1, 12))
    filtered_data = data[(data.index.month >= month[0]) & (data.index.month <= month[1])]

    #Plotting based on user input 
    if variable == "temperature":
        st.pyplot(plot_temp(filtered_data))
    elif variable == "percipitation":
        st.pyplot(plot_percipitation(filtered_data))
    elif variable == "wind speed":
        st.pyplot(plot_wind_speed(filtered_data))  
    elif variable == "wind gusts":
        st.pyplot(plot_wind_gusts(filtered_data))
    elif variable == "wind direction":
        st.pyplot(plot_wind_direction(filtered_data))
    elif variable == "All variables":
        st.pyplot(plot_all(filtered_data))   

elif options == 'Fun': #----------------------FUN PAGE------------------------------------
    st.title("Fun")
    st.write("Here's a fun fact: The highest temperature ever recorded on Earth was 134°F (56.7°C) in Death Valley, California!")



