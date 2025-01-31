# Add imports
import requests
import pandas as pd
import geopandas as gpd
import plotly.express as px
from datetime import datetime
import json
import os
from dotenv import load_dotenv

# Load EIA API key from environment
load_dotenv()
EIA_API_KEY = os.getenv('xVQk1mUsQQyWr6FMdfRQm5gn25VYlskXfVex0Pj5')

# Fetch energy data from EIA API
def fetch_eia_data():
    url = "https://api.eia.gov/v2/total-energy/data/"
    
    headers = {
        "X-Api-Key": EIA_API_KEY
    }
    
    params = {
        "frequency": "monthly",
        "data": ["value"],
        "sort": [
            {
                "column": "period",
                "direction": "desc"
            }
        ],
        "offset": 0,
        "length": 5000
    }
    try:
        response = requests.get(
            url, 
            headers=headers,
            params=json.dumps(params)
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API Request failed: {e}")
        return None

def process_energy_data(raw_data):
    if not raw_data:
        return pd.DataFrame()
        
    try:
        df = pd.DataFrame(raw_data.get('response', {}).get('data', []))
        if df.empty:
            return df
            
        df['year'] = pd.to_datetime(df['period'], format='%Y')
        df = df[['year', 'value', 'sectorName', 'stateDescription']]
        df = df[df['stateDescription'] == 'North Carolina']
        return df
    except Exception as e:
        print(f"Data processing error: {e}")
        return pd.DataFrame()

# Load NC county boundaries
nc_counties = gpd.read_file('NCCountyBoundaries.geojson')

# Get and process EIA data
energy_data = fetch_eia_data()
df_energy = process_energy_data(energy_data)

# Create visualization
fig = px.choropleth(
    df_energy,
    geojson=nc_counties,
    locations='stateDescription',
    featureidkey="properties.NAME",
    color='value',
    animation_frame='year',
    color_continuous_scale="Viridis",
    range_color=(0, df_energy['value'].max()),
    scope="usa",
    title='North Carolina Energy Consumption by Type (1970-2020)'
)

# Customize map view
fig.update_geos(
    fitbounds="locations",
    visible=False,
    center={"lat": 35.5, "lon": -80},
    scope='usa',
)

# Add energy type selector
fig.update_layout(
    updatemenus=[{
        'buttons': [
            {'method': 'update',
             'label': sector,
             'args': [{'z': [df_energy[df_energy['sectorName'] == sector]['value']]}]}
            for sector in df_energy['sectorName'].unique()
        ],
        'direction': 'down',
        'showactive': True,
    }]
)

fig.show()