import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from windrose import WindroseAxes
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

colors = {
    "temperature": "#C4611A",
    "precipitation": "#3173EE",
    "wind speed": "#AD4DE0",
    "wind gusts": "#3C1053",
    "wind direction": "#075E50",
}

series_colors = {
    "wind_gusts_10m": colors["wind gusts"],
    "wind_speed_10m": colors["wind speed"],
    "temperature_2m": colors["temperature"],
}

def plot_temp(dataframe):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(dataframe.index, dataframe['temperature_2m'], label='Temperature (°C)', color=colors["temperature"])
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
def plot_temp(dataframe, colors=None, width=800, height=500):
    # pick color for temperature line
    temp_color = colors["temperature"] if isinstance(colors, dict) and "temperature" in colors else "#1f77b4"

    xmin = dataframe.index.min()
    xmax = dataframe.index.max()

    fig = go.Figure()

    # Temperature line
    fig.add_trace(go.Scatter(
        x=dataframe.index,
        y=dataframe["temperature_2m"],
        mode="lines",
        name="Temperature (°C)",
        line=dict(color=temp_color)
    ))

    # Freezing point line (with legend)
    fig.add_trace(go.Scatter(
        x=[xmin, xmax],
        y=[0, 0],
        mode="lines",
        name="Freezing Point (0°C)",
        line=dict(color="#542F2F", dash="dash"),
        hoverinfo="skip"
    ))

    fig.update_layout(
        title="Temperature over time with freezing point",
        xaxis_title="Time",
        yaxis_title="Temperature (°C)",
        template="plotly_white",
        legend_title=None,
        width=width, 
        height=height
    )
    fig.update_xaxes(range=[xmin, xmax], showgrid=True)
    fig.update_yaxes(showgrid=True)

    return fig

#Plot the wind speed ----------------------------------------------------------------------------
def plot_wind_speed(dataframe):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(dataframe.index, dataframe['wind_speed_10m'], color=colors["wind speed"])

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
    ax.plot(dataframe.index, dataframe['wind_gusts_10m'], color=colors["wind gusts"])

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
    wind_dir = dataframe['wind_direction_10m']
    wind_spd = dataframe['wind_speed_10m']

    # Create a figure with two subplots: time series and windrose
    fig = plt.figure(figsize=(12, 8))

    # Subplot 1: Time series
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.scatter(dataframe.index, wind_dir, color=colors["wind direction"])
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
    daily_precip = dataframe['precipitation'].groupby(dataframe.index.date).sum()

    #calculating u and v components for wind direction arrows
    weekly_dir_rad = np.deg2rad(weekly_mean['wind_direction_10m'])
    u = np.cos(weekly_dir_rad)
    v = np.sin(weekly_dir_rad)

    #setting up the plot
    cols = ['wind_gusts_10m', 'wind_speed_10m', 'temperature_2m']


    fig, ax = plt.subplots(figsize=(10, 8))
    #plotting a horizontal line at 0 for oreintation
    ax.hlines(y=0, xmin=dataframe.index.min(), xmax=dataframe.index.max(), colors="#542F2F", linestyles='dashed')

    #plotting the weekly mean, min and max values of wind gusts, wind speed and temperature
    for idx, col in enumerate(cols):
        c = series_colors.get(col, "#4E6067")
        ax.plot(weekly_mean.index, weekly_mean[col], label=col, color=c)
        ax.fill_between(
            weekly_mean.index,
            weekly_min[col],
            weekly_max[col],
            alpha=0.2,
            color=c
        )
    #plotting daily precipitation as bars in the background
    ax.bar(daily_precip.index, daily_precip.values, color=colors["precipitation"], width=1.2)

    #adding wind direction arrows at the top of the plot
    ax.quiver(
        weekly_mean.index,
        [ax.get_ylim()[1]] * len(u),
        u, v,
        angles='xy', scale_units='xy', scale=0.35, color=colors["wind direction"], width=0.0015
    )

    # Custom legend handles
    wind_patch = mlines.Line2D([], [], color=colors["wind direction"], marker=r'$\rightarrow$', linestyle='None', markersize=12, label='Wind Direction')
    precip_patch = mpatches.Patch(color=colors["precipitation"], label="Precipitation (mm/day)")

    # Add to legend
    ax.legend(handles=[*ax.get_legend_handles_labels()[0], wind_patch, precip_patch], loc='lower center')

    #making the plot nice and readable
    ax.set_title('Weekly Weather Statistics with Daily Precipitation')
    ax.set_ylabel('Temperature (°C), Wind Speed/Gusts (m/s), Precipitation (mm)')
    ax.grid()
    ax.set_xlim([dataframe.index.min(), dataframe.index.max()])
    ax.set_xlabel('Time')   
    return fig