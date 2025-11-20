import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from windrose import WindroseAxes
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import sample_colorscale



# plot temperature with plotly --------------------------------------------------------------
def plot_temp(dataframe, colors=None, width=800, height=500):
    # pick color for temperature line
    from plotly import graph_objects as go
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

#Plot the temperaprecipitation as daily total as bar chart --------------------------------------------
def plot_precipitation(dataframe, colors=None, width=800, height=500):
    # Aggregate to daily totals (works best if index is a DatetimeIndex)
    from plotly import graph_objects as go
    if isinstance(dataframe.index, pd.DatetimeIndex):
        daily_precip = dataframe['precipitation'].resample('D').sum()
    else:
        daily_precip = dataframe['precipitation'].groupby(pd.to_datetime(dataframe.index).date).sum()
    daily_precip.index = pd.to_datetime(daily_precip.index)
    bar_color = (colors or {}).get("precipitation", "#1f77b4")
    one_day_ms = 24 * 60 * 60 * 1000

    fig = go.Figure(go.Bar(
        x=daily_precip.index,
        y=daily_precip.values,
        marker_color=bar_color,
        width=1.2 * one_day_ms,  # ~1.2 days wide to mirror matplotlib's width=1.2
        name="Daily precipitation"
    ))

    fig.update_traces(hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f} mm<extra></extra>")

    fig.update_layout(
        title="Daily Total Precipitation (mm)",
        xaxis_title="Time",
        yaxis_title="Total Precipitation (mm)",
        template="plotly_white",
        width=width,
        height=height
    )
    fig.update_xaxes(showgrid=False, tickangle=45)
    fig.update_yaxes(showgrid=True)

    return fig


#Plot the wind speed ----------------------------------------------------------------------------
def plot_wind_speed(dataframe, colors=None, width=800, height=500):
    import plotly.graph_objects as go
    wind_color = (colors or {}).get("wind speed", "#2ca02c")

    xmin = dataframe.index.min()
    xmax = dataframe.index.max()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dataframe.index,
        y=dataframe["wind_speed_10m"],
        mode="lines",
        name="Wind Speed (m/s)",
        line=dict(color=wind_color)
    ))

    fig.update_layout(
        title="Wind speed in m/s over time",
        xaxis_title="Time",
        yaxis_title="Wind Speed (m/s)",
        template="plotly_white",
        width=width,
        height=height,
        showlegend=False
    )
    fig.update_xaxes(range=[xmin, xmax], showgrid=True)
    fig.update_yaxes(range=[0, None], showgrid=True)

    return fig


#plot wind gusts --------------------------------------------------------------------------------
def plot_wind_gusts(dataframe, colors=None, width=800, height=500):
    import plotly.graph_objects as go
    gust_color = (colors or {}).get("wind gusts", "#d62728")

    xmin = dataframe.index.min()
    xmax = dataframe.index.max()

    fig = go.Figure(go.Scatter(
        x=dataframe.index,
        y=dataframe["wind_gusts_10m"],
        mode="lines",
        name="Wind gusts (m/s)",
        line=dict(color=gust_color)
    ))

    fig.update_traces(hovertemplate="%{x}<br>%{y:.2f} m/s<extra></extra>")

    fig.update_layout(
        title="Wind gusts in m/s over time",
        xaxis_title="Time",
        yaxis_title="Wind gusts (m/s)",
        template="plotly_white",
        width=width,
        height=height,
        showlegend=False
    )
    fig.update_xaxes(range=[xmin, xmax], showgrid=True)
    fig.update_yaxes(range=[0, None], showgrid=True)

    return fig

#plot the wind direction: over time and windrose -------------------------------------------------
def plot_wind_direction_plotly(
dataframe,
colors=None,
width=800,
height=800,
dir_bins=16,
speed_bins=None,    # e.g., [0, 2, 4, 6, 8, 10, float("inf")]
normalize=True      # match WindroseAxes(normed=True): percentages of all samples
):
    import plotly.graph_objects as go
    # Defaults
    scatter_color = (colors or {}).get("wind direction", "#9467bd")
    if speed_bins is None:
        speed_bins = [0, 2, 4, 6, 8, 10, float("inf")]

    # Prepare data
    df = dataframe.copy()
    df = df[["wind_direction_10m", "wind_speed_10m"]].dropna()
    if df.empty:
        # Return empty figure with titles
        fig = make_subplots(rows=2, cols=1, specs=[[{}], [{"type": "polar"}]],
                            row_heights=[0.45, 0.55], vertical_spacing=0.08)
        fig.update_layout(title="Wind Direction Over Time and Windrose", width=width, height=height)
        return fig

    # Convert index to datetime for time axis if needed
    try:
        x_time = pd.to_datetime(dataframe.index)
    except Exception:
        x_time = dataframe.index

    # Normalize directions to [0, 360)
    angles = np.mod(df["wind_direction_10m"].to_numpy(dtype=float), 360.0)
    speeds = df["wind_speed_10m"].to_numpy(dtype=float)

    # Direction bins (equal width)
    dir_edges = np.linspace(0, 360, dir_bins + 1)
    dir_centers = dir_edges[:-1] + (dir_edges[1] - dir_edges[0]) / 2.0
    dir_labels = np.arange(dir_bins)  # 0..dir_bins-1

    dir_cat = pd.cut(angles, bins=dir_edges, right=False, labels=dir_labels, include_lowest=True)
    spd_cat = pd.cut(speeds, bins=speed_bins, right=False)

    counts = pd.crosstab(dir_cat, spd_cat).reindex(index=dir_labels, fill_value=0)

    total = counts.values.sum()
    if normalize and total > 0:
        counts = counts / total * 100.0  # percentages
        r_suffix = " (%)"
    else:
        r_suffix = " (count)"

    # Build figure with two subplots
    fig = make_subplots(
        rows=2, cols=1,
        specs=[[{}], [{"type": "polar"}]],
        row_heights=[0.25, 0.75],
        vertical_spacing=0.17,
        subplot_titles=("Wind Direction Over Time", "Windrose - Wind Speed and Wind Direction")
    )
    fig.layout.annotations[1].update(y=0.65, x=0.50)
    # Subplot 1: time series scatter of wind direction
    fig.add_trace(
        go.Scatter(
            x=pd.to_datetime(dataframe.index),
            y=dataframe["wind_direction_10m"],
            mode="markers",
            name="Wind Direction (°)",
            marker=dict(color=scatter_color, size=5, opacity=0.6),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.0f}°<extra></extra>",
            showlegend=False
        ),
        row=1, col=1
    )


    # Colors for speed bins (Viridis discrete)
    n_traces = len(counts.columns)
    color_positions = np.linspace(0.15, 0.9, n_traces) if n_traces > 1 else [0.5]
    trace_colors = sample_colorscale("Viridis", color_positions)

    # Subplot 2: wind rose (stacked Barpolar per speed bin)
    width_deg = 360.0 / dir_bins
    for i, (col, color) in enumerate(zip(counts.columns, trace_colors)):
        r_vals = counts[col].values.astype(float)
        # Build friendly bin label
        left = col.left
        right = col.right
        if np.isinf(right):
            label = f">= {left:g}"
        else:
            label = f"{left:g} – {right:g}"
        fig.add_trace(
            go.Barpolar(
                r=r_vals,
                theta=dir_centers,
                width=[width_deg] * len(dir_centers),
                name=label,
                marker=dict(color=color, line=dict(color="#FFFFFF", width=1)),
                hovertemplate="Dir %{theta:.1f}°<br>" + f"{label} m/s<br>%{{r:.2f}}{r_suffix}<extra></extra>"
            ),
            row=2, col=1
        )

    # Layout and axes styling
    fig.update_layout(
        width=width,
        height=height,
        template="plotly_white",
        legend=dict(x=0.87, y=0.2),
        legend_title_text="Wind speed (m/s)",
        title_text="Wind Direction Over Time and Windrose"
    )

    # Time series axes
    fig.update_xaxes(title_text="Time", row=1, col=1, showgrid=True)
    fig.update_yaxes(title_text="Wind Direction (°)", row=1, col=1, range=[0, 360], showgrid=True, dtick=90)

    # Polar layout: 0° at North, clockwise (meteorological convention)
    fig.update_layout(
        polar=dict(
            angularaxis=dict(
                rotation=90,          # 0° at North
                direction="clockwise",
                showgrid=True,
                gridcolor="rgba(0,0,0,0.1)"
            ),
            radialaxis=dict(
                showgrid=True,
                gridcolor="rgba(0,0,0,0.1)",
                ticks="",
                angle=45
            )
        )
    )

    return fig


# Combined plot with weekly statistics and daily precipitation ------------------------------------------------------
def plot_all(
dataframe,
colors=None,          # e.g., {"precipitation": "#00a", "wind direction": "#a0a"}
series_colors=None,   # e.g., {"temperature": "#e24", "wind speed": "#28a", "wind gusts": "#f80"}
width=800,
height=500,
arrow_px=20           # constant pixel length for arrows
):
    import plotly.graph_objects as go
    colors = colors or {}
    series_colors = series_colors or {}

    alias = {
    "temperature_2m": "temperature",
    "wind_speed_10m": "wind speed",
    "wind_gusts_10m": "wind gusts",
    "wind_direction_10m": "wind direction",
    "precipitation": "precipitation",
    }

    # Ensure a DatetimeIndex
    idx = pd.to_datetime(dataframe.index)
    df = dataframe.copy()
    df.index = idx


    base_cols = ["wind_gusts_10m", "wind_speed_10m", "temperature_2m"]
    present_cols = [c for c in base_cols if c in df.columns]

    # Resampling rule: weekly anchored to weekday of first timestamp
    first_weekday = idx[0].strftime("%a").upper()[:3]
    rule = f"W-{first_weekday}"

    # Weekly stats
    weekly_mean = df.resample(rule).mean(numeric_only=True)
    weekly_min  = df.resample(rule).min(numeric_only=True)
    weekly_max  = df.resample(rule).max(numeric_only=True)

    # Daily precipitation
    has_precip = "precipitation" in df.columns
    if has_precip:
        daily_precip = df["precipitation"].resample("D").sum()
    else:
        daily_precip = pd.Series(dtype=float, index=pd.DatetimeIndex([]))

    # Wind direction arrows from weekly mean
    has_wdir = "wind_direction_10m" in weekly_mean.columns
    if has_wdir:
        weekly_dir_rad = np.deg2rad(weekly_mean["wind_direction_10m"].to_numpy(dtype=float))
        u = np.cos(weekly_dir_rad)
        v = np.sin(weekly_dir_rad)

    fig = go.Figure()

    # Precip bars (background)
    if has_precip and len(daily_precip):
        bar_color = colors.get("precipitation", "#3173EE")
        one_day_ms = 24 * 60 * 60 * 1000
        fig.add_trace(go.Bar(
            x=daily_precip.index,
            y=daily_precip.values,
            name="Precipitation (mm/day)",
            marker_color=bar_color,
            width=1.2 * one_day_ms,
            opacity=0.70
        ))

    # Helper: hex -> rgba
    def hex_to_rgba(h, a):
        h = h.lstrip("#")
        if len(h) == 3:
            h = "".join([c*2 for c in h])
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{a})"

    # Weekly bands and mean lines
    for col in present_cols:
        key = alias.get(col, col)  # friendly name if provided
        c = series_colors.get(key,  "#4E6067")

        # Upper bound (no legend)
        fig.add_trace(go.Scatter(
            x=weekly_max.index, y=weekly_max[col],
            mode="lines",
            line=dict(color=c, width=0),
            hoverinfo="skip",
            showlegend=False
        ))
        # Lower bound filled to previous (creates band)
        fig.add_trace(go.Scatter(
            x=weekly_min.index, y=weekly_min[col],
            mode="lines",
            line=dict(color=c, width=0),
            fill="tonexty",
            fillcolor=hex_to_rgba(c, 0.2),
            hoverinfo="skip",
            showlegend=False
        ))
        # Mean line
        fig.add_trace(go.Scatter(
            x=weekly_mean.index, y=weekly_mean[col],
            mode="lines",
            line=dict(color=c, width=2),
            name=key  # friendly label in legend
        ))

    # Horizontal dashed line at y=0
    xmin = idx.min()
    xmax = idx.max()
    fig.add_shape(
        type="line",
        x0=xmin, x1=xmax,
        y0=0, y1=0,
        xref="x", yref="y",
        line=dict(color="#542F2F", dash="dash")
    )

    # Compute y-range safely (avoid mixing 2D and 1D arrays)
    if present_cols:
        wk_max = np.nanmax(weekly_max[present_cols].to_numpy(dtype=float))
        wk_min = np.nanmin(weekly_min[present_cols].to_numpy(dtype=float))
    else:
        wk_max, wk_min = np.nan, np.nan

    if has_precip and len(daily_precip):
        prec_vals = daily_precip.to_numpy(dtype=float)
        dp_max = np.nanmax(prec_vals)
        dp_min = np.nanmin(prec_vals)
    else:
        dp_max, dp_min = np.nan, np.nan

    series_max = np.nanmax([wk_max, dp_max])
    series_min = np.nanmin([wk_min, 0.0, dp_min])

    if not np.isfinite(series_max) or not np.isfinite(series_min):
        series_max, series_min = 1.0, 0.0

    yrange = max(1e-9, series_max - series_min)
    y_top = series_max + 0.12 * yrange  # headroom for arrows
    y_bottom = series_min - 0.05 * yrange
    fig.update_yaxes(range=[y_bottom, y_top])

    # Wind direction arrows: constant pixel length
    if has_wdir and len(weekly_mean.index):
        wind_color = colors.get("wind direction", "#9467bd")
        t_idx = weekly_mean.index

        y_arrow = y_top - 0.04 * yrange  # arrow head y (constant)

        for i, t in enumerate(t_idx):
            if i >= len(u) or np.isnan(u[i]) or np.isnan(v[i]):
                continue
            # Arrow head at (t, y_arrow); tail is a fixed pixel offset away.
            fig.add_annotation(
                x=t, y=y_arrow,
                ax=-arrow_px * float(u[i]),   # pixel offsets -> same length for all
                ay=-arrow_px * float(v[i]),   # negative because +pixels in y go downward
                xref="x", yref="y",
                axref="pixel", ayref="pixel",
                showarrow=True,
                arrowhead=3, arrowsize=1, arrowwidth=2, arrowcolor=wind_color,
                opacity=0.9
            )

        # Legend entry for wind direction (dummy trace)
        fig.add_trace(go.Scatter(
            x=[xmin], y=[np.nan],
            mode="lines",
            line=dict(color=wind_color, width=2),
            name="Wind Direction (arrows)"
        ))

    # Layout
    fig.update_layout(
        width=width,
        height=height,
        template="plotly_white",
        bargap=0.05,
        legend=dict(orientation="h", yanchor="bottom", y=-0.30, x=0.5, xanchor="center"),
        title="Weekly Weather Statistics with Daily Precipitation",
        xaxis_title="Time",
        yaxis_title="Temperature (°C), Wind Speed/Gusts (m/s), Precipitation (mm)"
    )
    fig.update_xaxes(range=[xmin, xmax], showgrid=True)
    fig.update_yaxes(showgrid=True)

    return fig