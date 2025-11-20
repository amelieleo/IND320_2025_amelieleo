import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np


#---------------------------Pie chart----------------------------------------------
def create_pie_chart(df, title, color_map_production=None,
    threshold_pct=0.02, width=500, height=500, pull=0.07):
    import plotly.graph_objects as go
    # Validate input
    if not {"productiongroup", "quantitykwh"}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'productiongroup' and 'quantitykwh' columns.")
    color_map_production = color_map_production or {}

    # Aggregate
    pie_vals = df.groupby("productiongroup", dropna=False)["quantitykwh"].sum()
    total = pie_vals.sum()

    # Handle empty/zero
    if total == 0 or pie_vals.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"Production Distribution: {title}",
            width=width, height=height, template="plotly_white"
        )
        fig.add_annotation(text="No data", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig
    
    # Group small slices into "other"
    threshold = threshold_pct * total
    small = pie_vals[pie_vals < threshold]
    large = pie_vals[pie_vals >= threshold]

    main = large.copy()
    if not small.empty:
        main.loc["other"] = small.sum()

    labels = main.index.tolist()
    values = main.values.astype(float)

    # Colors (fallback gray for 'other', cycle for missing)
    default_other = "#9aa3ab"
    colors = [color_map_production.get(lbl, default_other if lbl == "other" else None) for lbl in labels]
    default_cycle = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    for i, c in enumerate(colors):
        if c is None:
            colors[i] = default_cycle[i % len(default_cycle)]

    pulls = [pull] * len(values)

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        pull=pulls,
        sort=False,
        marker=dict(colors=colors, line=dict(color="#FFFFFF", width=1)),
        textinfo="none",
        texttemplate="%{label}<br>%{percent:.2%}",
        hovertemplate="%{label}<br>%{value:,.0f} kWh<br>%{percent:.2%}<extra></extra>"
    ))

    fig.update_layout(
        title=f"Production Distribution: {title}",
        width=width,
        height=height,
        template="plotly_white",
        showlegend=True
    )

    return fig



#---------------------------line plot-------------------------------------------
def create_lineplot_production(df, title, color_map_production=None, width=800, height=500):
    import plotly.graph_objects as go

    color_map_production = {k.lower(): v for k, v in (color_map_production or {}).items()}

    # Transform data (same as your matplotlib version)
    df = df.copy()
    df["starttime"] = pd.to_datetime(df["starttime"])
    df = df.set_index("starttime").sort_index()
    time_data = df.groupby([df.index, "productiongroup"])["quantitykwh"].sum().unstack().fillna(0)

    fig = go.Figure()

    # Add one line per production group
    for label in time_data.columns:
        color = color_map_production.get(str(label).lower(), "#4E6067")
        fig.add_trace(go.Scatter(
            x=time_data.index,
            y=time_data[label],
            mode="lines",
            name=str(label),
            line=dict(color=color, width=2),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:,.0f} kWh<extra>%{fullData.name}</extra>"
        ))

    fig.update_layout(
        title=f"Hourly Electricity Production by Group ({title} Price Area)",
        xaxis_title="Time",
        yaxis_title="Quantity (kWh)",
        template="plotly_white",
        width=width,
        height=height,
        legend_title_text="Production Group"
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    return fig