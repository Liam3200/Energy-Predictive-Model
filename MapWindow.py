import sys
import os
import json
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
import matplotlib.pyplot as plt
import io
from folium import Choropleth, LayerControl, features
from branca.colormap import linear
from PyQt5.QtWidgets import *
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl, Qt
from PyQt5.QtGui import QPixmap
import sqlite3
from prophet import Prophet
from sklearn.linear_model import LinearRegression  

class EnergyMapWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('NC Energy Consumption Map')
        self.setGeometry(100, 100, 1200, 800)
        self.init_ui()
        self.load_data()
        self.ensure_all_years_exist()  # Add this line
        
        # Reload data after ensuring all years exist
        self.load_data()

    def init_ui(self):
        """Initialize the user interface components"""
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setSpacing(10)  # Add spacing between layout elements
        
        # Controls area
        controls_widget = QWidget()
        controls_widget.setMinimumHeight(100)
        controls_widget.setMaximumHeight(130)
        controls_layout = QHBoxLayout(controls_widget)
        controls_layout.setContentsMargins(10, 5, 10, 5)
        
        # Year selection widget
        year_widget = QWidget()
        year_layout = QVBoxLayout(year_widget)
        year_layout.setSpacing(8)
        
        # Year label
        year_label = QLabel("Year:")
        year_label.setAlignment(Qt.AlignCenter)
        year_label.setStyleSheet("font-weight: bold;")
        year_layout.addWidget(year_label)
        
        # Current year label
        self.year_value_label = QLabel()
        self.year_value_label.setAlignment(Qt.AlignCenter)
        self.year_value_label.setStyleSheet("font-weight: bold; font-size: 14px; margin: 5px 0;")
        year_layout.addWidget(self.year_value_label)
        
        # Add container for year buttons - centered now
        self.year_buttons_widget = QWidget()
        self.year_buttons_layout = QHBoxLayout(self.year_buttons_widget)
        self.year_buttons_layout.setContentsMargins(0, 0, 0, 0)
        self.year_buttons_layout.setAlignment(Qt.AlignCenter)
        year_layout.addWidget(self.year_buttons_widget)
        
        controls_layout.addWidget(year_widget)
        
        # Add vertical line separator
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setStyleSheet("color: #cccccc;")
        controls_layout.addWidget(line)
        
        # Rest of controls remain similar
        source_widget = QWidget()
        source_layout = QVBoxLayout(source_widget)
        source_layout.setSpacing(5)
        
        source_label = QLabel("Data Source:")
        source_label.setStyleSheet("font-weight: bold;")
        source_layout.addWidget(source_label)
        
        self.source_dropdown = QComboBox()
        self.source_dropdown.addItems(["All Data", "Historical Only", "Predictions Only"])
        self.source_dropdown.setMinimumWidth(150)
        self.source_dropdown.setStyleSheet("height: 25px;")
        source_layout.addWidget(self.source_dropdown)
        controls_layout.addWidget(source_widget)
        
        # Add another vertical line separator
        line2 = QFrame()
        line2.setFrameShape(QFrame.VLine)
        line2.setStyleSheet("color: #cccccc;")
        controls_layout.addWidget(line2)
        
        # Heating type dropdown
        heating_widget = QWidget()
        heating_layout = QVBoxLayout(heating_widget)
        heating_layout.setSpacing(5)
        
        heating_label = QLabel("Heating Type:")
        heating_label.setStyleSheet("font-weight: bold;")
        heating_layout.addWidget(heating_label)
        
        self.heating_dropdown = QComboBox()
        self.heating_dropdown.setMinimumWidth(200)
        self.heating_dropdown.setStyleSheet("height: 25px;")
        heating_layout.addWidget(self.heating_dropdown)
        controls_layout.addWidget(heating_widget)
        
        # Add vertical line separator for county filter
        line3 = QFrame()
        line3.setFrameShape(QFrame.VLine)
        line3.setStyleSheet("color: #cccccc;")
        controls_layout.addWidget(line3)
        
        # County selection (NEW)
        county_widget = QWidget()
        county_layout = QVBoxLayout(county_widget)
        county_layout.setSpacing(5)
        
        county_label = QLabel("County:")
        county_label.setStyleSheet("font-weight: bold;")
        county_layout.addWidget(county_label)
        
        self.county_dropdown = QComboBox()
        self.county_dropdown.setMinimumWidth(200)
        self.county_dropdown.setStyleSheet("height: 25px;")
        county_layout.addWidget(self.county_dropdown)
        controls_layout.addWidget(county_widget)
        
        # Add vertical line separator for view mode
        line4 = QFrame()
        line4.setFrameShape(QFrame.VLine)
        line4.setStyleSheet("color: #cccccc;")
        controls_layout.addWidget(line4)
        
        # View mode toggle (NEW)
        view_widget = QWidget()
        view_layout = QVBoxLayout(view_widget)
        view_layout.setSpacing(5)
        
        # Add a button for showing trends
        self.plot_btn = QPushButton("Show Trends")
        self.plot_btn.setMinimumWidth(120)
        self.plot_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border-radius: 4px;
                padding: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)
        view_layout.addWidget(self.plot_btn)
        controls_layout.addWidget(view_widget)
        
        # Add controls to main layout
        layout.addWidget(controls_widget)
        
        # Create a stacked widget to hold both map and plot views
        self.stacked_widget = QStackedWidget()
        layout.addWidget(self.stacked_widget)
        
        # Map view
        self.map_page = QWidget()
        map_layout = QVBoxLayout(self.map_page)
        self.web_view = QWebEngineView()
        self.web_view.setMinimumSize(800, 600)
        self.web_view.setStyleSheet("border: 1px solid #cccccc; border-radius: 4px;")
        map_layout.addWidget(self.web_view)
        self.stacked_widget.addWidget(self.map_page)
        
        # Plot view
        self.plot_page = QWidget()
        plot_layout = QVBoxLayout(self.plot_page)
        self.plot_view = QWebEngineView()
        self.plot_view.setMinimumSize(800, 600)
        self.plot_view.setStyleSheet("border: 1px solid #cccccc; border-radius: 4px;")
        plot_layout.addWidget(self.plot_view)
        self.stacked_widget.addWidget(self.plot_page)
        
        # Start with map view
        self.stacked_widget.setCurrentIndex(0)
        
        # Connect signals
        self.heating_dropdown.currentIndexChanged.connect(self.update_view)
        self.source_dropdown.currentIndexChanged.connect(self.update_view)
        self.county_dropdown.currentIndexChanged.connect(self.update_view)
        self.plot_btn.clicked.connect(self.toggle_view)

    def update_year_label(self):
        """Update the year label with current year"""
        if hasattr(self, 'current_year'):
            self.year_value_label.setText(f"Selected: {self.current_year}")
        else:
            self.year_value_label.setText("No year selected")

    def load_data(self):
        """Load data from SQLite database and county boundaries"""
        try:
            # Connect to database
            self.conn = sqlite3.connect('nc_energy.db')
            
            # Use local GeoJSON file
            nc_geo_path = os.path.join(os.path.dirname(__file__), "NCCountyBoundaries.geojson")
            
            # Check if file exists
            if not os.path.exists(nc_geo_path):
                print(f"GeoJSON file not found: {nc_geo_path}")
                QMessageBox.warning(self, "Warning", "County boundary file not found")
                self.county_geo = self.create_minimal_geojson()
            else:
                # Load the GeoJSON file
                try:
                    with open(nc_geo_path, 'r') as f:
                        self.county_geo = json.load(f)
                        print(f"Successfully loaded GeoJSON from {nc_geo_path}")
                        print(f"GeoJSON contains {len(self.county_geo['features'])} features")
                        
                        # Check and fix properties
                        self.fix_geojson_properties()
                        
                except Exception as geo_error:
                    print(f"Error loading GeoJSON: {geo_error}")
                    QMessageBox.warning(self, "Warning", f"Could not load county boundaries: {str(geo_error)}")
                    self.county_geo = self.create_minimal_geojson()
            
            # Load historical data
            try:
                historical_query = """
                SELECT County, Year, 
                       heated_by_electricity, heated_by_gas, 
                       heated_by_fuel_oil, heated_by_other, 
                       no_heating, heated_by_lp_gas
                FROM energy_consumption
                ORDER BY County, Year
                """
                self.historical_data = pd.read_sql_query(historical_query, self.conn)
                print(f"Loaded {len(self.historical_data)} historical records")
                if not self.historical_data.empty:
                    print(f"Sample historical data: {self.historical_data.head()}")
                    print(f"Historical years: {self.historical_data['Year'].unique()}")
            except Exception as e:
                print(f"Error loading historical data: {e}")
                self.historical_data = pd.DataFrame()
            
            # Load prediction data
            try:
                prediction_query = """
                SELECT County, Year, 
                       heated_by_electricity, heated_by_gas, 
                       heated_by_fuel_oil, heated_by_other, 
                       no_heating, heated_by_lp_gas
                FROM energy_predictions
                ORDER BY County, Year
                """
                self.prediction_data = pd.read_sql_query(prediction_query, self.conn)
                print(f"Loaded {len(self.prediction_data)} prediction records")
                if not self.prediction_data.empty:
                    print(f"Sample prediction data: {self.prediction_data.head()}")
                    print(f"Prediction years: {self.prediction_data['Year'].unique()}")
            except Exception as e:
                print(f"Error loading prediction data: {e}")
                self.prediction_data = pd.DataFrame()
            
            # Combine datasets
            self.all_data = pd.concat([self.historical_data, self.prediction_data])
            
            # Create sample data if no data exists
            if len(self.all_data) == 0:
                print("No data found. Creating sample data...")
                self.create_sample_data()
                
                # Try loading again
                self.historical_data = pd.read_sql_query(historical_query, self.conn)
                self.prediction_data = pd.read_sql_query(prediction_query, self.conn)
                self.all_data = pd.concat([self.historical_data, self.prediction_data])
                
                print(f"After creating sample data: {len(self.all_data)} records")
            
            # Update controls
            heating_types = [col for col in self.all_data.columns 
                            if col not in ['County', 'Year']]
            print(f"Heating types: {heating_types}")
            self.heating_dropdown.clear()  # Clear existing items
            self.heating_dropdown.addItems(heating_types)
            
            # Set year range
            if not self.all_data.empty:
                # Get all available years, including the specific years we want
                all_years = sorted(self.all_data['Year'].unique())
                
                if len(all_years) > 0:
                    min_year = int(all_years[0])
                    max_year = int(all_years[-1])
                    
                    # Define the specific years we want buttons for
                    specific_years = [1990, 2000, 2010, 2015, 2020, 2025, 2030, 2035, 2040]
                    
                    # Filter to only years in our data range
                    specific_years = [year for year in specific_years 
                                     if min_year <= year <= max_year]
                    
                    # Set initial current year
                    self.current_year = all_years[0]
                    self.update_year_label()
                    
                    # Create year buttons for specific years
                    self.create_year_buttons(specific_years, all_years)
                    
                    print(f"Data loaded successfully. Year range: {min_year}-{max_year}")
                    
                    # Update map initially
                    if heating_types:
                        self.heating_dropdown.setCurrentIndex(0)
                        self.update_map()
                else:
                    print("Warning: No years available in data")
                
            else:
                print("Warning: No data available in database")
                QMessageBox.warning(self, "Warning", "No data available in database")
            
            # After loading data, also populate county dropdown
            try:
                counties_query = "SELECT DISTINCT County FROM energy_consumption ORDER BY County"
                counties_df = pd.read_sql_query(counties_query, self.conn)
                counties = counties_df['County'].tolist()
                
                self.county_dropdown.clear()
                self.county_dropdown.addItems(counties)
                print(f"Loaded {len(counties)} counties")
            except Exception as e:
                print(f"Error loading counties: {e}")
            
        except Exception as e:
            import traceback
            print(f"Debug - Error details: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")

    def create_sample_data(self):
        """Create sample data if tables are empty"""
        try:
            # Sample NC counties
            counties = [
                'Alamance County', 'Alexander County', 'Alleghany County', 
                'Anson County', 'Ashe County', 'Avery County', 
                'Beaufort County', 'Bertie County', 'Bladen County', 
                'Brunswick County', 'Buncombe County', 'Burke County'
            ]
            
            # Historical years (1990-2020)
            hist_years = range(1990, 2025, 5)
            
            # Create historical data
            hist_data = []
            for county in counties:
                for year in hist_years:
                    # Generate random but plausible historical values
                    hist_data.append({
                        'County': county,
                        'Year': year,
                        'heated_by_electricity': int(800 + 100 * (year - 1990) + 300 * (0.5 - (year % 2))),
                        'heated_by_gas': int(500 + 80 * (year - 1990)),
                        'heated_by_fuel_oil': int(1000 - 40 * (year - 1990)),
                        'heated_by_other': int(300),
                        'no_heating': int(50),
                        'heated_by_lp_gas': int(500 - 15 * (year - 1990))
                    })
            
            # Prediction years (2025-2040)
            pred_years = range(2025, 2040, 5)
            
            # Create prediction data
            pred_data = []
            for county in counties:
                for year in pred_years:
                    # Generate random but increasing predicted values
                    pred_data.append({
                        'County': county,
                        'Year': year,
                        'heated_by_electricity': int(2000 + 300 * (year - 2025)),
                        'heated_by_gas': int(1500 + 150 * (year - 2025)),
                        'heated_by_fuel_oil': int(200 - 10 * (year - 2025)),
                        'heated_by_other': int(400),
                        'no_heating': int(100),
                        'heated_by_lp_gas': int(300 - 20 * (year - 2025))
                    })
            
            # Convert to DataFrames
            hist_df = pd.DataFrame(hist_data)
            pred_df = pd.DataFrame(pred_data)
            
            # Save to database
            hist_df.to_sql('energy_consumption', self.conn, if_exists='replace', index=False)
            pred_df.to_sql('energy_predictions', self.conn, if_exists='replace', index=False)
            
            print(f"Added {len(hist_df)} historical and {len(pred_df)} prediction records to database")
            
        except Exception as e:
            print(f"Error creating sample data: {str(e)}")

    def create_minimal_geojson(self, counties=None):
        """Create a minimal GeoJSON for NC counties"""
        print("Creating minimal GeoJSON for North Carolina counties")
        
        if not counties:
            counties = ["Alamance County", "Alexander County", "Alleghany County", 
                        "Anson County", "Ashe County"]
        
        features = []
        
        # Use a grid layout for counties
        cols = 10
        for i, county in enumerate(counties):
            # Strip 'County' suffix if present
            county_name = county.replace(' County', '')
            
            # Create grid layout
            row = i // cols
            col = i % cols
            
            # Calculate position
            lat = 36.0 - (row * 0.5)  # Start at top of NC
            lon = -84.0 + (col * 1.0)  # Start at west of NC
            
            # Create feature
            features.append({
                "type": "Feature",
                "properties": {"NAME": county_name},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[lon-0.4, lat-0.2], [lon+0.4, lat-0.2], 
                                     [lon+0.4, lat+0.2], [lon-0.4, lat+0.2],
                                     [lon-0.4, lat-0.2]]]
                }
            })
        
        return {"type": "FeatureCollection", "features": features}

    def fix_geojson_properties(self):
        """Ensure GeoJSON has the required properties"""
        print("Checking and fixing GeoJSON properties")
        feature_sample = self.county_geo['features'][0]['properties'] if self.county_geo['features'] else {}
        print(f"Sample feature properties: {feature_sample}")
        
        # Check if NAME property exists in features, if not, add it
        for feature in self.county_geo['features']:
            props = feature['properties']
            
            # Try to find county name from various property fields
            county_name = None
            for key in ['NAME', 'COUNTY', 'name', 'county', 'County', 'NAME10', 'NAMELSAD10']:
                if key in props:
                    county_name = props[key]
                    break
                    
            # If no county name found, try to extract from other properties
            if not county_name:
                # Just use the first property value as a fallback
                for key, value in props.items():
                    if isinstance(value, str) and len(value) > 0:
                        county_name = value
                        break
            
            # Set the NAME property explicitly
            if county_name:
                feature['properties']['NAME'] = county_name
        
        print(f"Fixed properties for {len(self.county_geo['features'])} features")
        
        # Verify we have the NAME property now
        count_with_name = sum(1 for f in self.county_geo['features'] if 'NAME' in f['properties'])
        print(f"Features with NAME property: {count_with_name} of {len(self.county_geo['features'])}")

    def update_map(self):
        """Update the choropleth map based on selected year and heating type"""
        try:
            # Use stored year instead of slider
            if not hasattr(self, 'current_year'):
                # If no year is selected yet, use first available year
                all_years = sorted(self.all_data['Year'].unique())
                if all_years:
                    self.current_year = all_years[0]
                else:
                    print("No years available")
                    return
            
            year = self.current_year
            heating_type = self.heating_dropdown.currentText()
            data_source = self.source_dropdown.currentText()
            
            print(f"Updating map for: Year={year}, Heating Type={heating_type}, Source={data_source}")
            
            if not heating_type:
                print("No heating type selected")
                return
            
            # Find closest available year if the exact year doesn't exist
            available_years = []
            if data_source == "Historical Only":
                available_years = sorted(self.historical_data['Year'].unique())
            elif data_source == "Predictions Only":
                available_years = sorted(self.prediction_data['Year'].unique())
            else:  # All Data
                available_years = sorted(self.all_data['Year'].unique())
            
            if not available_years:
                print(f"No data available for source: {data_source}")
                QMessageBox.warning(self, "No Data", f"No data available for {data_source}")
                return
                
            # Find closest year in available data
            closest_year = min(available_years, key=lambda x: abs(x - year))
            if closest_year != year:
                print(f"No data for year {year}, using closest available year: {closest_year}")
                year = closest_year
                # Update current year
                self.current_year = year
                self.update_year_label()
            
            # Get data for selected year
            if data_source == "Historical Only":
                df = self.historical_data[self.historical_data['Year'] == year]
                source_text = "Historical"
            elif data_source == "Predictions Only":
                df = self.prediction_data[self.prediction_data['Year'] == year]
                source_text = "Predicted"
            else:  # All Data
                if year <= self.historical_data['Year'].max() and not self.historical_data.empty:
                    df = self.historical_data[self.historical_data['Year'] == year]
                    source_text = "Historical"
                else:
                    df = self.prediction_data[self.prediction_data['Year'] == year]
                    source_text = "Predicted"
            
            # Check if data exists for selected year
            if df.empty:
                print(f"No {source_text.lower()} data available for {heating_type} in {year}")
                QMessageBox.warning(self, "No Data", 
                                   f"No {source_text.lower()} data available for {heating_type} in {year}")
                return
                
            print(f"Found {len(df)} records for year {year}")
            
            # Create a county name mapping dictionary
            county_name_map = {}
            for feature in self.county_geo['features']:
                county_name = feature['properties'].get('NAME', '')
                if county_name:
                    # Store mappings with and without 'County' suffix
                    county_name_map[county_name + ' County'] = county_name
                    county_name_map[county_name] = county_name
            
            print(f"Created mapping for {len(county_name_map)} counties")
            
            # Create base map centered on NC
            m = folium.Map(
                location=[35.5, -80], 
                zoom_start=7,
                tiles='CartoDB positron'
            )
            
            # Calculate percentages for each county
            # Get all heating types
            heating_types = [col for col in df.columns 
                           if col not in ['County', 'Year']]
            
            # Copy all heating type data for calculating percentages
            df_all_types = df[['County'] + heating_types].copy()
            
            # Calculate total energy usage across all types for each county
            df_all_types['total_energy'] = df_all_types[heating_types].sum(axis=1)
            
            # Calculate percentage for the selected heating type
            df_all_types['percentage'] = (df_all_types[heating_type] / df_all_types['total_energy'] * 100).round(1)
            
            # Extract selected heating type data with percentage
            df_subset = df[['County', heating_type]].copy()
            df_subset = df_subset.rename(columns={heating_type: 'value'})
            
            # Add percentage to df_subset by merging
            df_subset = df_subset.merge(
                df_all_types[['County', 'percentage']], 
                on='County', 
                how='left'
            )
            
            print(f"Value range: {df_subset['value'].min()} to {df_subset['value'].max()}")
            print(f"Percentage range: {df_subset['percentage'].min()}% to {df_subset['percentage'].max()}%")
            
            # Normalize county names for matching with GeoJSON
            # First remove ' County' suffix if present
            df_subset['GeoJSON_County'] = df_subset['County'].str.replace(' County', '', regex=False)
            
            # Print mapped values for debugging
            print("County mappings (sample):")
            for i, (orig, mapped) in enumerate(zip(df_subset['County'], df_subset['GeoJSON_County'])):
                if i < 5:  # Just show first 5 for brevity
                    print(f"  {orig} → {mapped}")
            
            # Check match rate
            geojson_counties = set(feature['properties'].get('NAME', '') for feature in self.county_geo['features'])
            matched_counties = df_subset['GeoJSON_County'].isin(geojson_counties)
            match_count = matched_counties.sum()
            print(f"Counties matched with GeoJSON: {match_count} of {len(df_subset)}")
            
            if match_count == 0:
                # If no matches, try a simplified approach using basic county map
                self.county_geo = self.create_minimal_geojson(counties=df_subset['County'].tolist())
                print("Using minimal GeoJSON due to no county matches")
                
                # Reset mappings
                df_subset['GeoJSON_County'] = df_subset['County'].str.replace(' County', '', regex=False)
                
            # Use a simpler approach with GeoJSON that we know is compatible
            # Create simplified GeoJSON with our data
            simplified_geo = {"type": "FeatureCollection", "features": []}
            
            for _, row in df_subset.iterrows():
                county = row['GeoJSON_County']
                value = row['value']
                percentage = row['percentage']
                
                # Find matching feature in original GeoJSON
                matching_feature = None
                for feature in self.county_geo['features']:
                    if feature['properties'].get('NAME', '') == county:
                        matching_feature = feature
                        break
                
                if matching_feature:
                    # Create a copy with our data
                    new_feature = {
                        "type": "Feature",
                        "properties": {
                            "NAME": county,
                            "value": float(value),
                            "percentage": float(percentage)
                        },
                        "geometry": matching_feature['geometry']
                    }
                    simplified_geo['features'].append(new_feature)
                else:
                    print(f"No geometry found for {county}")
            
            # Use simplified choropleth approach
            if len(simplified_geo['features']) > 0:
                # Add choropleth directly using simplified GeoJSON
                choropleth = folium.GeoJson(
                    simplified_geo,
                    name='choropleth',
                    style_function=lambda feature: {
                        'fillColor': self._get_color(feature['properties']['value'], 
                                                min(f['properties']['value'] for f in simplified_geo['features']),
                                                max(f['properties']['value'] for f in simplified_geo['features'])),
                        'color': 'black',
                        'weight': 1,
                        'fillOpacity': 0.7
                    }
                ).add_to(m)
                
                # Add enhanced tooltips with percentage
                folium.GeoJsonTooltip(
                    fields=['NAME', 'value', 'percentage'],
                    aliases=['County:', f'{heating_type}:', 'Percentage of Total:'],
                    labels=True,
                    localize=True,
                    sticky=False,
                    style='''
                        background-color: white; 
                        color: #333333; 
                        font-family: arial; 
                        font-size: 12px; 
                        padding: 10px;
                        border-radius: 3px;
                        box-shadow: 0 1px 5px rgba(0,0,0,0.4);
                    ''',
                    toLocaleString=True
                ).add_to(choropleth)
                
                # Add a legend
                colormap = folium.LinearColormap(
                    colors=['#ffffb2', '#fecc5c', '#fd8d3c', '#f03b20', '#bd0026'],
                    index=[min(f['properties']['value'] for f in simplified_geo['features']),
                           max(f['properties']['value'] for f in simplified_geo['features'])],
                    vmin=min(f['properties']['value'] for f in simplified_geo['features']),
                    vmax=max(f['properties']['value'] for f in simplified_geo['features']),
                    caption=f'{heating_type} ({year})'
                )
                colormap.add_to(m)
            
            # Add title
            title_html = f'''
                <h3 style="text-align:center;font-family:Arial;margin-bottom:5px;">
                    NC Energy Consumption - {heating_type} ({year})
                </h3>
                <p style="text-align:center;font-family:Arial;font-size:14px;margin-top:0;">
                    {source_text} Data
                </p>
            '''
            m.get_root().html.add_child(folium.Element(title_html))
            
            # Save to HTML and load in web view
            temp_path = os.path.join(os.path.dirname(__file__), "temp_map.html")
            m.save(temp_path)
            
            # Load in web view
            self.web_view.setUrl(QUrl.fromLocalFile(os.path.abspath(temp_path)))
            self.web_view.reload()  # Force reload
            
            print("Map updated successfully")
            
        except Exception as e:
            import traceback
            print(f"Debug - Error details: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            QMessageBox.warning(self, "Error", f"Failed to update map: {str(e)}")

    def _get_color(self, value, vmin, vmax):
        """Return color based on value"""
        if vmax == vmin:
            # Avoid division by zero
            normalized = 0.5
        else:
            normalized = (value - vmin) / (vmax - vmin)
        
        colors = ['#ffffb2', '#fecc5c', '#fd8d3c', '#f03b20', '#bd0026']
        idx = min(int(normalized * len(colors)), len(colors) - 1)
        return colors[idx]

    def create_year_buttons(self, years, available_years):
        """Create clickable buttons for each year"""
        # Clear any existing buttons
        while self.year_buttons_layout.count():
            item = self.year_buttons_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Convert available years to a set for faster lookups
        available_years_set = set(available_years)
        
        # Now that we don't have a slider, make buttons larger and more prominent
        btn_width = 65
        btn_height = 32
        
        # Create button for each year
        for year in years:
            btn = QPushButton(str(year))
            
            # Set fixed width and height - larger now
            btn.setFixedWidth(btn_width)
            btn.setFixedHeight(btn_height)
            
            # Use larger font since buttons are now the primary control
            font = btn.font()
            font.setPointSize(10)
            btn.setFont(font)
            
            # Check if year exists in data
            if year in available_years_set:
                # Year has exact data
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #4CAF50;
                        color: white;
                        border-radius: 4px;
                        padding: 4px 2px;
                        text-align: center;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #45a049;
                    }
                    QPushButton:pressed {
                        background-color: #3e8e41;
                    }
                """)
            else:
                # No data for this year
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #f1f1f1;
                        color: #666;
                        border-radius: 4px;
                        padding: 4px 2px;
                        text-align: center;
                    }
                    QPushButton:hover {
                        background-color: #ddd;
                    }
                    QPushButton:pressed {
                        background-color: #ccc;
                    }
                """)
            
            # Connect button signal to year selection
            btn.clicked.connect(lambda checked, y=year: self.select_year(y))
            self.year_buttons_layout.addWidget(btn)

    def select_year(self, year):
        """Set the selected year and update map"""
        print(f"Year button clicked: {year}")
        
        # Get available years based on data source
        data_source = self.source_dropdown.currentText()
        available_years = []
        
        if data_source == "Historical Only":
            available_years = sorted(self.historical_data['Year'].unique())
        elif data_source == "Predictions Only":
            available_years = sorted(self.prediction_data['Year'].unique())
        else:  # All Data
            available_years = sorted(self.all_data['Year'].unique())
        
        if not available_years:
            return
        
        # Find closest available year
        if year in available_years:
            closest_year = year
        else:
            closest_year = min(available_years, key=lambda y: abs(y - year))
        
        # Store the selected year
        self.current_year = closest_year
        
        # Update the year label
        self.update_year_label()
        
        # Update the map
        self.update_map()

    def ensure_all_years_exist(self):
        """Ensure all required years exist in the database"""
        try:
            # Get existing years
            query = "SELECT DISTINCT Year FROM energy_predictions ORDER BY Year"
            existing_years = pd.read_sql_query(query, self.conn)['Year'].tolist()
            
            # Define required years
            required_years = [1990, 2000, 2010, 2015, 2020, 2025, 2030, 2035, 2040]
            
            # Find missing years
            missing_years = [year for year in required_years if year not in existing_years]
            
            if not missing_years:
                print("All required years exist in the database")
                return
                
            print(f"Adding missing years to database: {missing_years}")
            
            # For each missing year, interpolate data from surrounding years
            for missing_year in missing_years:
                # Find surrounding years
                lower_years = [y for y in existing_years if y < missing_year]
                higher_years = [y for y in existing_years if y > missing_year]
                
                if not lower_years or not higher_years:
                    print(f"Cannot interpolate for {missing_year} - no surrounding years")
                    continue
                    
                lower_year = max(lower_years)
                higher_year = min(higher_years)
                
                # Get data for surrounding years
                query = f"""
                SELECT County, 
                       heated_by_electricity, heated_by_gas, 
                       heated_by_fuel_oil, heated_by_other,
                       no_heating, heated_by_lp_gas
                FROM energy_predictions 
                WHERE Year IN ({lower_year}, {higher_year})
                ORDER BY County, Year
                """
                
                surrounding_data = pd.read_sql_query(query, self.conn)
                
                # Calculate weight for interpolation
                weight = (missing_year - lower_year) / (higher_year - lower_year)
                
                # Process each county
                counties = surrounding_data['County'].unique()
                new_records = []
                
                for county in counties:
                    county_data = surrounding_data[surrounding_data['County'] == county]
                    
                    if len(county_data) != 2:
                        print(f"Skipping {county} - incomplete data")
                        continue
                        
                    lower_data = county_data.iloc[0]
                    higher_data = county_data.iloc[1]
                    
                    # Interpolate values
                    new_record = {
                        'County': county,
                        'Year': missing_year
                    }
                    
                    for column in ['heated_by_electricity', 'heated_by_gas', 
                                  'heated_by_fuel_oil', 'heated_by_other',
                                  'no_heating', 'heated_by_lp_gas']:
                        lower_val = lower_data[column]
                        higher_val = higher_data[column]
                        interpolated = int(lower_val + weight * (higher_val - lower_val))
                        new_record[column] = interpolated
                    
                    new_records.append(new_record)
                
                # Insert new records into database
                if new_records:
                    new_df = pd.DataFrame(new_records)
                    new_df.to_sql('energy_predictions', self.conn, if_exists='append', index=False)
                    print(f"Added {len(new_records)} records for year {missing_year}")
        
        except Exception as e:
            print(f"Error ensuring all years exist: {e}")

    def toggle_view(self):
        """Toggle between map and trend plot views"""
        current_index = self.stacked_widget.currentIndex()
        
        if current_index == 0:  # Currently map view
            # Verify dependencies before switching to plot view
            try:
                from prophet import Prophet
            except ImportError:
                QMessageBox.warning(self, "Missing Dependency", 
                                  "Prophet library is required for forecasting. Please install prophet package.")
                return
                
            try:
                import plotly.graph_objects
            except ImportError:
                QMessageBox.warning(self, "Missing Dependency", 
                                  "Plotly library is required for visualization. Please install plotly package.")
                return
                
            # Make sure a county is selected
            if not self.county_dropdown.currentText():
                QMessageBox.warning(self, "Selection Required", "Please select a county to view trends")
                return
                
            # Switch to plot view
            self.plot_btn.setText("Show Map")
            self.plot_btn.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    border-radius: 4px;
                    padding: 6px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
            """)
            self.stacked_widget.setCurrentIndex(1)
            self.update_plot()
        else:  # Currently plot view
            # Switch to map view
            self.plot_btn.setText("Show Trends")
            self.plot_btn.setStyleSheet("""
                QPushButton {
                    background-color: #2196F3;
                    color: white;
                    border-radius: 4px;
                    padding: 6px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #0b7dda;
                }
            """)
            self.stacked_widget.setCurrentIndex(0)
            self.update_map()

    def update_view(self):
        """Update the current view based on selected options"""
        if self.stacked_widget.currentIndex() == 0:
            self.update_map()
        else:
            self.update_plot()

    def update_plot(self):
        """Generate and display Prophet prediction plot for selected county and heating type"""
        try:
            county = self.county_dropdown.currentText()
            heating_type = self.heating_dropdown.currentText()
            
            if not county or not heating_type:
                print("No county or heating type selected")
                QMessageBox.warning(self, "Warning", "Please select both a county and heating type")
                return
            
            print(f"Generating trend plot for {county}, {heating_type}")
            
            # Show wait cursor
            QApplication.setOverrideCursor(Qt.WaitCursor)
            
            # Get all available data for this county and heating type
            query = f"""
            WITH combined AS (
                SELECT County, Year, {heating_type} as value, 'Historical' as source
                FROM energy_consumption
                WHERE County = ?
                UNION ALL
                SELECT County, Year, {heating_type} as value, 'Prediction' as source
                FROM energy_predictions
                WHERE County = ?
            )
            SELECT * FROM combined
            ORDER BY Year
            """
            
            df = pd.read_sql_query(query, self.conn, params=[county, county])
            
            if df.empty:
                print(f"No data available for {county}, {heating_type}")
                QMessageBox.warning(self, "No Data", f"No data available for {county}, {heating_type}")
                QApplication.restoreOverrideCursor()
                return
            
            # Prepare data for Prophet
            df['Year'] = df['Year'].astype(int)
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            
            # Create dataframe for Prophet
            prophet_df = pd.DataFrame({
                'ds': pd.to_datetime(df['Year'].astype(str) + '-01-01'),
                'y': df['value'],
                'source': df['source']
            })
            
            # Only use historical data for training
            historical_df = prophet_df[df['source'] == 'Historical']
            
            if len(historical_df) < 2:
                QMessageBox.warning(self, "Insufficient Data", 
                                  "Not enough historical data points for forecasting.")
                QApplication.restoreOverrideCursor()
                return
            
            try:
                import plotly.graph_objects as go
                import plotly.io as pio
                from prophet import Prophet
                import numpy as np
                
                # Create model with desired parameters
                model = Prophet(
                    yearly_seasonality=False,
                    growth='linear',
                    changepoint_prior_scale=0.05,
                    interval_width=0.9  # 90% confidence interval
                )
                
                # Fit the model with historical data
                model.fit(historical_df[['ds', 'y']])
                
                # Create future dataframe to include historical period and future
                future = pd.DataFrame({
                    'ds': pd.date_range(
                        start=f'{df["Year"].min()}-01-01',
                        end='2040-01-01',
                        freq='YS'
                    )
                })
                
                # Make predictions with Prophet model
                forecast = model.predict(future)
                
                # Create a custom plotly figure
                fig = go.Figure()
                
                # Get data by source type
                prediction_df = df[df['source'] == 'Prediction'].copy()
                historical_df = df[df['source'] == 'Historical'].copy()
                
                # Get the last historical year for boundary
                last_historical_year = historical_df['Year'].max() if not historical_df.empty else None
                
                # Get dates and values for historical and prediction data
                hist_dates = [pd.Timestamp(f"{year}-01-01") for year in sorted(historical_df['Year'].unique())]
                hist_values = [historical_df[historical_df['Year'] == year]['value'].iloc[0] for year in sorted(historical_df['Year'].unique())]
                
                pred_dates = [pd.Timestamp(f"{year}-01-01") for year in sorted(prediction_df['Year'].unique())]
                pred_values = [prediction_df[prediction_df['Year'] == year]['value'].iloc[0] for year in sorted(prediction_df['Year'].unique())]
                
                # Start uncertainty from the last historical data point
                if not hist_dates:
                    print("No historical dates available")
                    QMessageBox.warning(self, "No Historical Data", "No historical data available for this county and heating type.")
                    QApplication.restoreOverrideCursor()
                    return
                    
                # Get last historical point
                last_hist_date = hist_dates[-1]
                last_hist_value = hist_values[-1]
                
                # Combine for uncertainty visualization (starting from last historical point)
                combined_dates = [last_hist_date] + pred_dates
                combined_values = [last_hist_value] + pred_values
                
                # Improved uncertainty calculation with guaranteed visibility
                def calculate_improved_uncertainty(base_value, position_index, mean_value, is_declining=False):
                    """Calculate uncertainty that ensures visibility even for small values"""
                    # Ensure we have a reasonable base to work with
                    base = max(base_value, 1)  # Prevent divide by zero errors
                    mean = max(mean_value, 1)  # For scaling purpose
                    
                    # Initialize variables to ensure they're always defined
                    scale_factor = 0.05  # Default value
                    base_min_uncertainty = max(10, mean * 0.05)  # Default minimum uncertainty
                    
                    # Special handling for fuel oil and LP gas
                    if heating_type in ['heated_by_fuel_oil', 'heated_by_lp_gas']:
                        # Higher base uncertainty for these types
                        base_min_uncertainty = max(20, mean * 0.20)  # At least 20% of mean
                        
                        # Different scales based on position
                        if position_index == 0:
                            scale_factor = 0.05  # Start with 5% for historical endpoint
                        else:
                            # More conservative growth if values are declining toward zero
                            if is_declining and base_value < 50:
                                # For declining values approaching zero, use fixed absolute uncertainty
                                # This prevents the band from collapsing as values approach zero
                                fixed_uncertainty = max(50, mean * 0.10)
                                return fixed_uncertainty
                            else:
                                # Standard growth for normal cases
                                scale_factor = 0.10 + 0.10 * position_index + 0.02 * (position_index ** 2)
                                scale_factor = min(0.8, scale_factor)  # Cap at 80%
                    else:
                        # For other heating types (electricity, gas, etc.)
                        base_min_uncertainty = max(10, mean * 0.02)  # At least 2% of mean value
                        
                        if position_index == 0:
                            scale_factor = 0.01  # Very small for the historical endpoint
                        else:
                            # Standard growth rate
                            scale_factor = 0.05 + 0.05 * position_index + 0.01 * (position_index ** 2)
                            scale_factor = min(0.5, scale_factor)  # Cap at 50%
                    
                    # Calculate uncertainty - use both absolute and relative components
                    relative_component = base * scale_factor
                    absolute_component = base_min_uncertainty * (1 + position_index * 0.2)
                    
                    # Use the larger of the two to ensure visibility
                    uncertainty = max(relative_component, absolute_component)
                    
                    return uncertainty
                
                # Calculate mean value for scaling purposes
                mean_value = np.mean(combined_values)
                
                # Detect if we have a declining trend toward zero
                is_declining_to_zero = False
                if len(combined_values) > 2:
                    # Check if at least the last 2 points are decreasing and final value is small
                    if (combined_values[-1] < combined_values[-2] and 
                        combined_values[-1] < 50):
                        is_declining_to_zero = True
                
                # Calculate bounds with enhanced visibility for all heating types
                upper_bounds = []
                lower_bounds = []
                for i, (point_date, point_value) in enumerate(zip(combined_dates, combined_values)):
                    # Check if this point is part of a declining sequence
                    is_in_decline = False
                    if i > 0 and point_value < combined_values[i-1]:
                        is_in_decline = True
                    
                    # Get uncertainty that's guaranteed to be visible
                    uncertainty = calculate_improved_uncertainty(point_value, i, mean_value, is_in_decline)
                    
                    # Set bounds ensuring they're reasonable and visible
                    upper_bound = point_value + uncertainty
                    lower_bound = max(0, point_value - uncertainty)  # Ensure non-negative
                    
                    # If we have previous points, smooth transitions
                    if i > 0:
                        prev_upper = upper_bounds[-1]
                        prev_lower = lower_bounds[-1]
                        
                        # Special handling for LP gas and fuel oil with near-zero values
                        if heating_type in ['heated_by_fuel_oil', 'heated_by_lp_gas']:
                            if is_declining_to_zero:
                                # For declining trends toward zero, maintain a minimum upper bound
                                # This ensures the uncertainty region doesn't collapse
                                min_upper = max(50, mean_value * 0.15)
                                upper_bound = max(upper_bound, min_upper, point_value * 1.5)
                                
                                # Extremely permissive for declining trends (allow larger changes)
                                max_change_factor = 3.0
                            else:
                                # Normal case
                                max_change_factor = 2.0
                        elif point_value < 100:
                            max_change_factor = 1.5  # More permissive for small values
                        else:
                            max_change_factor = 1.3  # Standard smoothing
                        
                        # Always ensure point is within bounds with significant margin
                        upper_bound = max(upper_bound, point_value * 1.20)  # At least 20% above point
                        
                        # For near-zero values, use more generous lower bound
                        if point_value < 20:
                            lower_bound = 0  # Always use zero for very small values
                        else:
                            lower_bound = min(lower_bound, point_value * 0.80)  # At least 20% below
                        
                        # Apply smoothing - but less restrictive for declining trends
                        if is_declining_to_zero and i > 1:
                            # Different approach for declining trends - prioritize not collapsing
                            # Don't constrain the upper bound too much when values approach zero
                            upper_bound = max(upper_bound, prev_upper * 0.7)  # Allow at most 30% decrease
                        else:
                            # Standard smoothing for normal cases
                            upper_bound = min(upper_bound, prev_upper * max_change_factor)
                        
                        lower_bound = max(lower_bound, prev_lower / max_change_factor, 0)

                    # Extra special handling for fuel oil and LP gas near zero values
                    if (heating_type in ['heated_by_fuel_oil', 'heated_by_lp_gas'] and point_value < 30) or point_value < 20:
                        lower_bound = 0  # Always start at zero
                        
                        # Set minimum visible band even for zero values
                        # More aggressive for last few points to ensure visibility
                        if i >= len(combined_dates) - 3 and point_value < 10:
                            # For final points approaching zero, keep a visible band
                            upper_bound = max(60, mean_value * 0.15)
                        else:
                            # Standard minimum for small/zero values
                            upper_bound = max(50, point_value * 3.0)

                    # More aggressive minimum width for declining trends
                    band_width = upper_bound - lower_bound
                    if heating_type in ['heated_by_fuel_oil', 'heated_by_lp_gas']:
                        if is_declining_to_zero:
                            # Ensure extra wide bands for declining trends
                            min_width = max(50, mean_value * 0.3)
                        else:
                            min_width = max(30, mean_value * 0.25)
                        
                        if band_width < min_width:
                            upper_bound = lower_bound + min_width
                    elif band_width < 10:
                        upper_bound = lower_bound + max(10, mean_value * 0.10)
                    
                    upper_bounds.append(upper_bound)
                    lower_bounds.append(lower_bound)
                
                # Add uncertainty bands - start from the last historical point now
                fig.add_trace(
                    go.Scatter(
                        x=combined_dates,  # Include the last historical point
                        y=upper_bounds,
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=combined_dates,  # Include the last historical point
                        y=lower_bounds,
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(65, 105, 225, 0.3)',
                        name='Prediction Interval'
                    )
                )
                
                # Add historical data points with connecting line
                fig.add_trace(
                    go.Scatter(
                        x=hist_dates,
                        y=hist_values,
                        mode='markers+lines',
                        name='Historical Data',
                        line=dict(color='green', width=2),
                        marker=dict(color='green', size=10),
                        hovertemplate='%{x|%Y}: %{y:,.0f} units<extra></extra>'
                    )
                )
                
                # Add prediction data points with connecting line
                fig.add_trace(
                    go.Scatter(
                        x=pred_dates,
                        y=pred_values,
                        mode='markers+lines',
                        name='Predictions',
                        line=dict(color='blue', width=2),
                        marker=dict(color='blue', size=10),
                        hovertemplate='%{x|%Y}: %{y:,.0f} units<extra></extra>'
                    )
                )
                
                # Add connecting line between historical and prediction data
                if hist_dates and pred_dates:
                    last_hist_date = hist_dates[-1]
                    first_pred_date = pred_dates[0]
                    last_hist_value = hist_values[-1]
                    first_pred_value = pred_values[0]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=[last_hist_date, first_pred_date],
                            y=[last_hist_value, first_pred_value],
                            mode='lines',
                            line=dict(color='gray', width=2, dash='dot'),
                            showlegend=False
                        )
                    )
                
                # Format the plot title with proper capitalization
                formatted_heating_type = heating_type.replace('heated_by_', '').replace('_', ' ').title()
                
                # Customize layout
                fig.update_layout(
                    title=f'{county} - {formatted_heating_type} Energy Usage Forecast (1990-2040)',
                    xaxis_title='Year',
                    yaxis_title='Number of Housing Units',
                    hovermode='x unified',
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01,
                        bgcolor='rgba(255, 255, 255, 0.8)'
                    ),
                    template='plotly_white',
                    annotations=[
                        dict(
                            x=0.5,
                            y=0,
                            xref="paper",
                            yref="paper",
                            text="Uncertainty increases with forecast distance",
                            showarrow=False,
                            font=dict(size=12, color="gray", style="italic"),
                            xanchor="center",
                            yanchor="top",
                            yshift=-30
                        )
                    ]
                )
                
                # Format x-axis to show years only
                fig.update_xaxes(
                    tickformat="%Y",
                    dtick="M24",  # Every 2 years to avoid crowding
                    tickangle=45,
                    tickmode='linear'
                )
                
                # Add vertical line separating historical from prediction
                if last_historical_year:
                    divider_date = pd.Timestamp(f"{last_historical_year}-12-31")
                    
                    fig.add_shape(
                        type="line",
                        x0=divider_date,
                        y0=0,
                        x1=divider_date,
                        y1=1,
                        yref="paper",
                        line=dict(
                            color="gray",
                            width=2,
                            dash="dash",
                        )
                    )
                    
                    fig.add_annotation(
                        x=divider_date,
                        y=1,
                        yref="paper",
                        text="Historical | Predicted",
                        showarrow=False,
                        font=dict(size=12),
                        xanchor="center",
                        yanchor="bottom"
                    )
                
                # Save to temporary HTML file
                temp_path = os.path.join(os.path.dirname(__file__), "prophet_plot.html")
                pio.write_html(fig, file=temp_path, auto_open=False)
                
                # Display the plot
                self.plot_view.setUrl(QUrl.fromLocalFile(os.path.abspath(temp_path)))
                self.plot_view.reload()
                
                print("Prophet visualization created successfully")
                
            except Exception as model_error:
                import traceback
                print(f"Error in model generation: {str(model_error)}")
                print(f"Traceback: {traceback.format_exc()}")
                QMessageBox.warning(self, "Model Error", 
                                 f"Error generating forecast model: {str(model_error)}")
            
            QApplication.restoreOverrideCursor()
            
        except Exception as e:
            QApplication.restoreOverrideCursor()  # Always restore cursor
            import traceback
            print(f"Error generating plot: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            QMessageBox.warning(self, "Error", f"Failed to generate plot: {str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = EnergyMapWindow()
    window.show()
    sys.exit(app.exec_())

