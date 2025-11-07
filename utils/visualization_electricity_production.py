import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np

color_map_production = {
    'hydro':   '#0072B2',  # Blue
    'thermal': "#C75702",  # Vermilion
    'wind':    '#009E73',  # Bluish green
    'solar':   '#E69F00',  # Orange (dunkler als Gelb, besser auf Weiß)
    'other':   '#6C757D'   # Neutral gray
}

#---------------------------Pie chart----------------------------------------------
def create_pie_chart(df, title):
    
    # Define threshold percentage for grouping small categories
    threshold_pct = 0.02
    pie_vals = df.groupby('productiongroup')['quantitykwh'].sum()
    threshold = threshold_pct * pie_vals.sum()
    small_slices = pie_vals[pie_vals < threshold]
    large_slices = pie_vals[pie_vals >= threshold]

    # Create a new category 'other' for all small slices
    main_pie_data = large_slices
    if not small_slices.empty:
        main_pie_data['other'] = small_slices.sum()
    
    colors = [color_map_production[label] for label in main_pie_data.index]
    
    explode = np.ones(len(main_pie_data)) * 0.07
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.pie(
        main_pie_data.values,
        labels=main_pie_data.index,
        autopct='%1.2f%%',
        explode=explode,
        colors=colors  
    )
    ax.set_title(f'Production Distribution: {title}')
    return fig

#---------------------------line plot-------------------------------------------
def create_lineplot_production(df, title):
    #transforming the data
    df = df.set_index('starttime').sort_index()
    time_data = df.groupby([df.index, 'productiongroup'])['quantitykwh'].sum().unstack().fillna(0)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    for label in time_data.columns:
        ax.plot(
            time_data.index,
            time_data[label],
            label=label,
            color=color_map_production.get(label.lower(), "#4E6067")
        )
    ax.set_title(f'Hourly Electricity Production by Group ({title} Price Area)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Quantity (kWh)')
    ax.grid()
    ax.legend(title='Production Group')

    return fig