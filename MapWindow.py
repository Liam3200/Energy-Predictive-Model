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
            
            # Generate Prophet model
            try:
                model_data = self.generate_prophet_model(df, county, heating_type)
                
                if model_data is not None:
                    # Use the new display_plotly_forecast method
                    self.display_plotly_forecast(model_data, county, heating_type)
                else:
                    QMessageBox.warning(self, "Forecast Error", "Failed to generate forecast model")
            except Exception as model_error:
                print(f"Error in model generation: {str(model_error)}")
                QMessageBox.warning(self, "Model Error", f"Error generating forecast model: {str(model_error)}")
                
            QApplication.restoreOverrideCursor()
            
        except Exception as e:
            QApplication.restoreOverrideCursor()  # Always restore cursor
            print(f"Error generating plot: {str(e)}")
            QMessageBox.warning(self, "Error", f"Failed to generate plot: {str(e)}")

    def generate_prophet_model(self, df, county, heating_type):
        """Generate Prophet model and forecast using population data with uncertainty visualization"""
        try:
            # Ensure Prophet library is available
            try:
                from prophet import Prophet
            except ImportError:
                QMessageBox.warning(self, "Missing Dependency", 
                                  "Prophet library is required for forecasting. Please install prophet package.")
                return None
            
            # First, get ALL energy types for this county to understand distribution
            all_energy_query = """
            SELECT Year, 
                   heated_by_electricity, heated_by_gas, heated_by_fuel_oil, 
                   heated_by_other, no_heating, heated_by_lp_gas
            FROM energy_consumption
            WHERE County = ?
            ORDER BY Year
            """
            energy_df = pd.read_sql_query(all_energy_query, self.conn, params=[county])
            
            if energy_df.empty:
                print(f"No energy data found for {county}")
                return None
                
            # Get population data for this county
            population_query = """
            SELECT Population_1990, Population_2000, Population_2010, Population_2020
            FROM county_populations
            WHERE County = ? OR County = ? OR County LIKE ?
            """
            
            base_county = county.replace(' County', '')
            pop_df = pd.read_sql_query(population_query, self.conn, 
                                      params=[county, base_county, f"{base_county}%"])
            
            # Calculate minimum values based on historical data to prevent unrealistically low values
            min_values = {}
            for col in ['heated_by_electricity', 'heated_by_gas', 'heated_by_fuel_oil', 
                      'heated_by_other', 'no_heating', 'heated_by_lp_gas']:
                # Get non-zero values
                non_zero_values = energy_df[energy_df[col] > 0][col]
                if not non_zero_values.empty:
                    # Set minimum to either 5% of the mean non-zero value or 1, whichever is larger
                    min_values[col] = max(int(non_zero_values.mean() * 0.05), 1)
                else:
                    min_values[col] = 1
            
            # Get population data (create synthetic if needed)
            if pop_df.empty:
                print(f"No population data found for {county}")
                # Create synthetic population based on housing units
                energy_df['Population'] = None
                # Use NC state average population growth rates as fallback
                nc_growth_rates = {
                    1990: 0.8,   # base
                    2000: 0.9,   # ~12% growth from 1990
                    2010: 1.0,   # ~18% growth from 2000
                    2020: 1.1,   # ~10% growth from 2010
                    2025: 1.15,  # projected
                    2030: 1.2,   # projected
                    2035: 1.25,  # projected
                    2040: 1.3    # projected
                }
                
                # Create synthetic population based on total housing and state growth
                base_year = energy_df['Year'].min()
                base_housing = energy_df.loc[energy_df['Year'] == base_year, 
                                          ['heated_by_electricity', 'heated_by_gas', 'heated_by_fuel_oil', 
                                           'heated_by_other', 'no_heating', 'heated_by_lp_gas']].sum(axis=1).iloc[0]
                
                energy_df['Population'] = energy_df['Year'].apply(
                    lambda y: base_housing * 2.5 * nc_growth_rates.get(y, 1) if y in nc_growth_rates else None
                )
                population_years = {y: p for y, p in zip(energy_df['Year'], energy_df['Population'])}
            else:
                # Create population series with years as index
                population_years = {
                    1990: pop_df['Population_1990'].iloc[0] if not pd.isna(pop_df['Population_1990'].iloc[0]) else None,
                    2000: pop_df['Population_2000'].iloc[0] if not pd.isna(pop_df['Population_2000'].iloc[0]) else None,
                    2010: pop_df['Population_2010'].iloc[0] if not pd.isna(pop_df['Population_2010'].iloc[0]) else None,
                    2020: pop_df['Population_2020'].iloc[0] if not pd.isna(pop_df['Population_2020'].iloc[0]) else None
                }
                
                # Add population to energy dataframe
                energy_df['Population'] = energy_df['Year'].map(population_years)
            
            # Fill any missing population values
            energy_df = energy_df.sort_values('Year')
            energy_df['Population'] = energy_df['Population'].interpolate().ffill().bfill()
            
            # Calculate total housing units and percentages
            energy_df['total_housing'] = energy_df[['heated_by_electricity', 'heated_by_gas', 'heated_by_fuel_oil', 
                                                'heated_by_other', 'no_heating', 'heated_by_lp_gas']].sum(axis=1)
            
            # Calculate housing per capita
            energy_df['housing_per_capita'] = energy_df['total_housing'] / energy_df['Population']
            energy_df['housing_per_capita'] = energy_df['housing_per_capita'].replace([np.inf, -np.inf], np.nan)
            energy_df['housing_per_capita'] = energy_df['housing_per_capita'].fillna(energy_df['housing_per_capita'].median())
            
            # Determine decline rate thresholds based on energy type
            decline_thresholds = {
                'heated_by_electricity': 0,  # No decline floor for electricity (growing)
                'heated_by_gas': 0.3,        # At least 30% of latest value for gas
                'heated_by_fuel_oil': 0.4,   # At least 40% of latest value for fuel oil
                'heated_by_other': 0.4,      # At least 40% of latest value for other
                'no_heating': 0.5,           # At least 50% of latest value for no heating
                'heated_by_lp_gas': 0.3      # At least 30% of latest value for LP gas
            }
            
            # Get latest year data
            latest_year = energy_df['Year'].max()
            latest_data = energy_df[energy_df['Year'] == latest_year]
            
            # Project future population based on available data
            pop_df = energy_df[['Year', 'Population']].dropna()
            
            if len(pop_df) >= 2:
                # Calculate average annual growth rate
                first_year = pop_df['Year'].min()
                last_year = pop_df['Year'].max()
                first_pop = pop_df.loc[pop_df['Year'] == first_year, 'Population'].iloc[0]
                last_pop = pop_df.loc[pop_df['Year'] == last_year, 'Population'].iloc[0]
                years_diff = last_year - first_year
                
                if years_diff > 0 and first_pop > 0:
                    # Calculate compound annual growth rate
                    annual_growth_rate = (last_pop / first_pop) ** (1 / years_diff) - 1
                    
                    # Ensure growth rate is realistic (between -0.5% and 3% per year)
                    annual_growth_rate = max(-0.005, min(annual_growth_rate, 0.03))
                    
                    # Project future populations
                    latest_pop = pop_df.loc[pop_df['Year'] == pop_df['Year'].max(), 'Population'].iloc[0]
                    future_pop = {year: latest_pop * (1 + annual_growth_rate) ** (year - last_year) 
                               for year in range(1990, 2041)}
                else:
                    # Fallback to state average growth
                    latest_pop = pop_df.loc[pop_df['Year'] == pop_df['Year'].max(), 'Population'].iloc[0]
                    future_pop = {
                        2025: latest_pop * 1.05,
                        2030: latest_pop * 1.10,
                        2035: latest_pop * 1.15,
                        2040: latest_pop * 1.20
                    }
            else:
                # Simple growth for missing data
                if latest_data.iloc[0]['Population'] > 0:
                    latest_pop = latest_data.iloc[0]['Population']
                else:
                    # Estimate from housing if population data is missing
                    latest_pop = latest_data.iloc[0]['total_housing'] * 2.5
                    
                future_pop = {
                    2025: latest_pop * 1.05,
                    2030: latest_pop * 1.10,
                    2035: latest_pop * 1.15,
                    2040: latest_pop * 1.20
                }
                
            # Prepare data for Prophet from input dataframe
            # Ensure proper data types
            df['Year'] = df['Year'].astype(int)
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            
            # Replace missing values with median
            if df['value'].isnull().any():
                median_val = df['value'].median()
                df['value'] = df['value'].fillna(median_val)
            
            # Create Prophet dataframe
            prophet_df = pd.DataFrame({
                'ds': pd.to_datetime(df['Year'].astype(str) + '-01-01'),  # January 1st for each year
                'y': df['value'],
                'population': df['Year'].map(lambda y: 
                                   population_years.get(y, future_pop.get(y, latest_pop * 1.1))
                               )
            })
            
            # Fill any missing population values
            prophet_df['population'] = prophet_df['population'].interpolate().ffill().bfill()
            
            # Create Prophet model
            model = Prophet(
                yearly_seasonality=False,
                growth='linear',
                changepoint_prior_scale=0.05,  # More conservative
                interval_width=0.9  # 90% confidence interval
            )
            
            # Add population as a regressor
            model.add_regressor('population')
            
            # Fit model
            try:
                model.fit(prophet_df)
            except Exception as e:
                print(f"Error fitting Prophet model: {str(e)}")
                return None
            
            # Create future dataframe for all years
            future = pd.DataFrame({
                'ds': pd.date_range(start='1990-01-01', end='2040-12-31', freq='YS')
            })
            
            # Add population to future dataframe
            future['population'] = future['ds'].dt.year.map(
                lambda y: population_years.get(y, future_pop.get(y, latest_pop * 1.1))
            )
            
            # Make Prophet predictions
            forecast = model.predict(future)
            
            # Determine the latest value and apply decline thresholds
            latest_value = None
            for idx, row in df.iterrows():
                if row['Year'] == latest_year:
                    latest_value = row['value']
                    break
                    
            if latest_value is None and len(df) > 0:
                latest_value = df.iloc[-1]['value']
                
            # If we still don't have a value, use the predicted value for the latest year
            if latest_value is None:
                latest_mask = forecast['ds'].dt.year == latest_year
                if latest_mask.any():
                    latest_value = forecast.loc[latest_mask, 'yhat'].iloc[0]
                else:
                    latest_value = min_values.get(heating_type, 1)
            
            # Apply increasing uncertainty for future years
            base_uncertainty = 0.1  # 10% for the first future year
            for year in range(2025, 2041):
                year_mask = forecast['ds'].dt.year == year
                if year_mask.any():
                    # Calculate years from latest historical data
                    years_out = year - latest_year
                    
                    # Increase uncertainty by 5% for each 5 years into the future
                    uncertainty_factor = 1.0 + (base_uncertainty * (years_out / 5))
                    
                    # Apply to confidence intervals - widen them proportionally
                    idx = forecast.index[year_mask][0]
                    mean_val = forecast.loc[idx, 'yhat']
                    lower_diff = mean_val - forecast.loc[idx, 'yhat_lower'] 
                    upper_diff = forecast.loc[idx, 'yhat_upper'] - mean_val
                    
                    # Apply wider intervals
                    forecast.loc[idx, 'yhat_lower'] = mean_val - (lower_diff * uncertainty_factor)
                    forecast.loc[idx, 'yhat_upper'] = mean_val + (upper_diff * uncertainty_factor)
                    
                    # For declining energy types, ensure we respect the minimum thresholds
                    if heating_type in decline_thresholds and heating_type != 'heated_by_electricity':
                        threshold = decline_thresholds[heating_type]
                        # Apply progressively stronger floors for further years
                        year_factor = max(0.1, 1 - (0.2 * (years_out / 5)))  # Gradual decline
                        floor_value = latest_value * threshold * year_factor
                        
                        # Apply the floor, but ensure it's at least min_values[heating_type]
                        min_val = max(floor_value, min_values.get(heating_type, 1))
                        forecast.loc[idx, 'yhat'] = max(forecast.loc[idx, 'yhat'], min_val)
                        forecast.loc[idx, 'yhat_lower'] = max(forecast.loc[idx, 'yhat_lower'], min_val * 0.8)
            
            # Ensure predictions are non-negative
            forecast['yhat'] = np.maximum(forecast['yhat'], 0)
            forecast['yhat_lower'] = np.maximum(forecast['yhat_lower'], 0)
            forecast['yhat_upper'] = np.maximum(forecast['yhat_upper'], min_values.get(heating_type, 0))
            
            # Round values to integers since we're working with housing units
            forecast['yhat'] = forecast['yhat'].round().astype(int)
            forecast['yhat_lower'] = forecast['yhat_lower'].round().astype(int)
            forecast['yhat_upper'] = forecast['yhat_upper'].round().astype(int)
            
            # Add actual historical data
            forecast_with_history = forecast.copy()
            forecast_with_history['actual'] = None
            forecast_with_history['source'] = None
            
            # Map actual values to forecast
            for _, row in df.iterrows():
                year = int(row['Year'])
                value = float(row['value'])
                source = str(row['source'])
                
                mask = forecast_with_history['ds'].dt.year == year
                if mask.any():
                    idx = forecast_with_history.index[mask][0]
                    forecast_with_history.loc[idx, 'actual'] = value
                    forecast_with_history.loc[idx, 'source'] = source
            
            return forecast_with_history
            
        except Exception as e:
            import traceback
            print(f"Error in Prophet model: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            return None

    def display_plotly_forecast(self, forecast, county, heating_type):
        """Display the Prophet forecast using Plotly with accurate uncertainty visualization"""
        try:
            # Ensure plotly library is available
            try:
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
            except ImportError:
                QMessageBox.warning(self, "Missing Dependency", 
                                 "Plotly library is required for visualization. Please install plotly package.")
                return
            
            # Create figure
            fig = make_subplots(specs=[[{"secondary_y": False}]])
            
            # Ensure forecast has expected columns
            required_cols = ['ds', 'yhat', 'yhat_upper', 'yhat_lower']
            if not all(col in forecast.columns for col in required_cols):
                print(f"Missing columns in forecast data: {[col for col in required_cols if col not in forecast.columns]}")
                QMessageBox.warning(self, "Data Error", "Incomplete forecast data - missing required columns")
                return
            
            # Convert data to proper types
            forecast['yhat'] = pd.to_numeric(forecast['yhat'], errors='coerce')
            forecast['yhat_upper'] = pd.to_numeric(forecast['yhat_upper'], errors='coerce')
            forecast['yhat_lower'] = pd.to_numeric(forecast['yhat_lower'], errors='coerce')
            
            # Split forecast into historical and future periods
            historical_mask = forecast['actual'].notna()
            
            # Get the last year of historical data
            try:
                # Get the last year with historical data
                last_historical_year = forecast.loc[historical_mask, 'ds'].dt.year.max()
                print(f"Last historical year: {last_historical_year}")
            except:
                # If there's an error, use 2020 as a fallback
                last_historical_year = 2020
                print(f"Using fallback historical year: {last_historical_year}")
            
            # Create prediction mask for points after the last historical year
            prediction_mask = forecast['ds'].dt.year > last_historical_year
            
            # Add historical data points
            if historical_mask.any():
                historical_data = forecast[historical_mask]
                fig.add_trace(
                    go.Scatter(
                        x=historical_data['ds'],
                        y=historical_data['actual'],
                        mode='markers+lines',
                        name='Historical Data',
                        line=dict(color='green', width=3),
                        marker=dict(size=8, color='green')
                    )
                )
            
            # Separate the forecast display
            if prediction_mask.any():
                prediction_data = forecast[prediction_mask]
                
                # Add main prediction line
                fig.add_trace(
                    go.Scatter(
                        x=prediction_data['ds'],
                        y=prediction_data['yhat'],
                        mode='lines',
                        name='Forecast',
                        line=dict(color='royalblue', width=2)
                    )
                )
                
                # Add uncertainty bands - only for prediction period
                # Upper bound
                fig.add_trace(
                    go.Scatter(
                        x=prediction_data['ds'],
                        y=prediction_data['yhat_upper'],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False
                    )
                )
                
                # Lower bound with fill
                fig.add_trace(
                    go.Scatter(
                        x=prediction_data['ds'],
                        y=prediction_data['yhat_lower'],
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(65, 105, 225, 0.2)',
                        name='Prediction Interval'
                    )
                )
            
            # Add any actual prediction points that exist in the database
            prediction_data_mask = (forecast['source'] == 'Prediction') & forecast['actual'].notna()
            if prediction_data_mask.any():
                pred_points = forecast[prediction_data_mask]
                fig.add_trace(
                    go.Scatter(
                        x=pred_points['ds'],
                        y=pred_points['actual'],
                        mode='markers',
                        name='Prediction Data Points',
                        marker=dict(size=8, color='orange')
                    )
                )
            
            # Calculate the last data point for smoothing the transition
            if historical_mask.any():
                last_historical = forecast[historical_mask].iloc[-1]
                first_prediction_idx = forecast[prediction_mask].index[0] if prediction_mask.any() else None
                
                # Add a transition line between historical and prediction
                if first_prediction_idx is not None:
                    first_prediction = forecast.loc[first_prediction_idx]
                    transition_x = [last_historical['ds'], first_prediction['ds']]
                    transition_y = [last_historical['actual'], first_prediction['yhat']]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=transition_x,
                            y=transition_y,
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
                dtick="M12",  # Every 12 months
                tickangle=45
            )
            
            # Add vertical line separating historical from prediction - FIXED VERSION
            # Instead of creating a new timestamp, we'll use shape objects
            if historical_mask.any() and prediction_mask.any():
                # Find the last historical data point and first prediction point
                last_historical_date = forecast[historical_mask]['ds'].max()
                first_prediction_date = forecast[prediction_mask]['ds'].min()
                
                # Use the midpoint between the two dates as the dividing line
                separator_date = last_historical_date
                
                # Add a shape as a vertical line
                fig.add_shape(
                    type="line",
                    x0=separator_date,
                    y0=0,
                    x1=separator_date,
                    y1=1,
                    yref="paper",
                    line=dict(
                        color="gray",
                        width=2,
                        dash="dash",
                    )
                )
                
                # Add an annotation for the line
                fig.add_annotation(
                    x=separator_date,
                    y=1,
                    yref="paper",
                    text="Historical | Predicted",
                    showarrow=False,
                    font=dict(size=12),
                    xanchor="center",
                    yanchor="bottom"
                )
            
            # Save to HTML and display
            plot_path = os.path.join(os.path.dirname(__file__), "temp_plot.html")
            fig.write_html(plot_path)
            
            # Load in web view
            self.plot_view.setUrl(QUrl.fromLocalFile(os.path.abspath(plot_path)))
            self.plot_view.reload()
            
            print("Plot generated successfully")
            
        except Exception as e:
            import traceback
            print(f"Error creating plot: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            QMessageBox.warning(self, "Error", f"Failed to create plot: {str(e)}")

    def plot_time_series(self, df, county, energy_type):
        """
        Plot time series data with uncertainty bounds for future predictions
        """
        if df is None or df.empty:
            return
        
        plt.figure(figsize=(10, 6))
        
        # Determine where historical data ends and predictions begin
        historical_mask = ~df['actual'].isna()
        prediction_mask = ~historical_mask
        
        # Split data into historical and predictions
        historical = df[historical_mask]
        predictions = df[prediction_mask]
        
        # Plot historical data
        historical_years = historical['ds'].dt.year
        historical_values = historical['actual']
        plt.plot(historical_years, historical_values, 'o-', color='blue', label='Historical Data')
        
        # Plot predictions with uncertainty
        prediction_years = predictions['ds'].dt.year
        prediction_values = predictions['yhat']
        prediction_lower = predictions['yhat_lower']
        prediction_upper = predictions['yhat_upper']
        
        # Plot the main prediction line
        plt.plot(prediction_years, prediction_values, '--', color='red', label='Predictions')
        
        # Plot uncertainty bounds with transparency
        plt.fill_between(prediction_years, prediction_lower, prediction_upper, 
                        color='red', alpha=0.2, label='Prediction Interval (90%)')
        
        # Add title and labels
        title = f"{energy_type.replace('heated_by_', '').replace('_', ' ').title()} Usage in {county}"
        plt.title(title)
        plt.xlabel('Year')
        plt.ylabel('Number of Housing Units')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend
        plt.legend()
        
        # Format x-axis to show years
        plt.xticks(range(1990, 2045, 5))
        
        # Enhance readability
        plt.tight_layout()
        
        # Save the figure to a BytesIO object
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close()
        
        # Convert to QPixmap and display
        buf.seek(0)
        pixmap = QPixmap()
        pixmap.loadFromData(buf.getvalue())
        self.plot_label.setPixmap(pixmap)
        self.plot_label.setScaledContents(True)

    def display_prophet_plot(self, model, forecast, county, heating_type):
        """Display the Prophet model's built-in plot with uncertainty visualization"""
        try:
            # Ensure required libraries are available
            try:
                import matplotlib.pyplot as plt
                from prophet.plot import plot_components, plot_cross_validation_metric
                import io
                from PyQt5.QtGui import QPixmap
                import tempfile
                import os
            except ImportError as e:
                QMessageBox.warning(self, "Missing Dependency", 
                                  f"Required library is missing: {e}")
                return
                
            # Create a temporary file for the plot
            with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
                temp_path = f.name
            
            # Use Prophet's built-in plot_plotly method which returns a plotly figure
            try:
                # First, create the plot using Prophet's built-in capability
                from prophet.plot import plot_plotly, plot_components_plotly
                
                # Generate plotly figure with Prophet's built-in method
                fig = plot_plotly(model, forecast, uncertainty=True, plot_cap=False, 
                                include_legend=True, figsize=(1000, 600))
                
                # Customize the figure
                formatted_heating_type = heating_type.replace('heated_by_', '').replace('_', ' ').title()
                fig.update_layout(
                    title=f'{county} - {formatted_heating_type} Energy Usage Forecast (1990-2040)',
                    xaxis_title='Year',
                    yaxis_title='Number of Housing Units',
                    hovermode='x unified',
                    template='plotly_white'
                )
                
                # Save the plot to a temporary HTML file
                import plotly.offline as py
                py.plot(fig, filename=temp_path, auto_open=False)
                
                # Load the HTML file in the web view
                self.plot_view.setUrl(QUrl.fromLocalFile(os.path.abspath(temp_path)))
                self.plot_view.reload()
                
                print("Prophet plot generated successfully using plotly")
                return
                
            except Exception as plotly_error:
                print(f"Error using Prophet's plotly output: {str(plotly_error)}")
                print("Falling back to matplotlib output")
                
                # Fallback to matplotlib if plotly version fails
                try:
                    # Create a figure
                    plt.figure(figsize=(12, 8))
                    
                    # Use Prophet's built-in plotting with matplotlib
                    model.plot(forecast, uncertainty=True, figsize=(12, 8))
                    
                    # Customize the plot
                    formatted_heating_type = heating_type.replace('heated_by_', '').replace('_', ' ').title()
                    plt.title(f'{county} - {formatted_heating_type} Energy Usage Forecast (1990-2040)')
                    plt.xlabel('Year')
                    plt.ylabel('Number of Housing Units')
                    plt.grid(True, linestyle='--', alpha=0.7)
                    
                    # Save the figure to a temporary file
                    plt.savefig(temp_path, format='png', dpi=100)
                    plt.close()
                    
                    # Load in web view using HTML
                    html_content = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>Prophet Forecast</title>
                        <style>
                            body {{ margin: 0; padding: 0; text-align: center; }}
                            img {{ max-width: 100%; height: auto; }}
                        </style>
                    </head>
                    <body>
                        <img src="file://{temp_path}" alt="Prophet Forecast">
                    </body>
                    </html>
                    """
                    
                    # Save HTML to temporary file
                    html_path = temp_path + '.html'
                    with open(html_path, 'w') as f:
                        f.write(html_content)
                    
                    # Load the HTML in the web view
                    self.plot_view.setUrl(QUrl.fromLocalFile(os.path.abspath(html_path)))
                    self.plot_view.reload()
                    
                    print("Prophet plot generated successfully using matplotlib")
                    
                except Exception as mpl_error:
                    print(f"Error generating matplotlib plot: {str(mpl_error)}")
                    QMessageBox.warning(self, "Plot Error", 
                                      f"Failed to generate Prophet plot: {str(mpl_error)}")
                    return
                
        except Exception as e:
            import traceback
            print(f"Error in display_prophet_plot: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            QMessageBox.warning(self, "Error", f"Failed to display Prophet plot: {str(e)}")

def create_prophet_predictions(conn, county_name, heating_type):
    """
    Generate predictions for a county and heating type using population growth to
    inform housing demand and maintain realistic energy type distributions.
    """
    # Get base county name (without 'County' suffix if present)
    base_county_name = county_name.replace(' County', '')
    
    # First, get ALL energy types for this county to understand distribution patterns
    all_energy_query = """
    SELECT e.Year, 
           e.heated_by_electricity, e.heated_by_gas, e.heated_by_fuel_oil, 
           e.heated_by_other, e.no_heating, e.heated_by_lp_gas
    FROM energy_consumption e
    WHERE e.County = ?
    ORDER BY e.Year
    """
    energy_df = pd.read_sql_query(all_energy_query, conn, params=[county_name])
    
    if energy_df.empty:
        print(f"No energy data found for {county_name}")
        return None, None
    
    # Get population data separately for better control
    pop_query = """
    SELECT County, Population_1990, Population_2000, Population_2010, Population_2020
    FROM county_populations
    WHERE County = ? OR County = ? OR County LIKE ?
    """
    pop_df = pd.read_sql_query(pop_query, conn, 
                             params=[county_name, base_county_name, f"{base_county_name}%"])
    
    if pop_df.empty:
        # Try a more fuzzy approach to find closest match
        all_counties = pd.read_sql_query("SELECT County FROM county_populations", conn)
        
        for idx, pop_county in all_counties.iterrows():
            if base_county_name.lower() in pop_county['County'].lower() or \
               pop_county['County'].lower() in base_county_name.lower():
                print(f"Found potential match: {pop_county['County']} for {county_name}")
                pop_df = pd.read_sql_query(
                    "SELECT * FROM county_populations WHERE County = ?", 
                    conn, params=[pop_county['County']]
                )
                break
    
    # Create population dictionary by year
    population_by_year = {}
    
    if not pop_df.empty:
        # Extract population values from population table
        if not pd.isna(pop_df['Population_1990'].iloc[0]):
            population_by_year[1990] = float(pop_df['Population_1990'].iloc[0])
        if not pd.isna(pop_df['Population_2000'].iloc[0]):
            population_by_year[2000] = float(pop_df['Population_2000'].iloc[0])
        if not pd.isna(pop_df['Population_2010'].iloc[0]):
            population_by_year[2010] = float(pop_df['Population_2010'].iloc[0])
        if not pd.isna(pop_df['Population_2020'].iloc[0]):
            population_by_year[2020] = float(pop_df['Population_2020'].iloc[0])
    
    # If no population data was found, create synthetic population
    if not population_by_year:
        print(f"No population data found for {county_name}, using synthetic population")
        # Calculate total housing units for each year
        energy_df['total_housing'] = energy_df[['heated_by_electricity', 'heated_by_gas', 'heated_by_fuel_oil', 
                                             'heated_by_other', 'no_heating', 'heated_by_lp_gas']].sum(axis=1)
        
        # Estimate population based on housing units (assume ~2.5 people per housing unit)
        for idx, row in energy_df.iterrows():
            population_by_year[row['Year']] = row['total_housing'] * 2.5
    
    # Add population column to energy_df
    energy_df['Population'] = energy_df['Year'].map(population_by_year)
    
    # Fill any remaining NaN values with interpolation and extrapolation
    # First sort by year to ensure proper interpolation
    energy_df = energy_df.sort_values('Year')
    
    # Use interpolation to fill gaps
    energy_df['Population'] = energy_df['Population'].interpolate(method='linear')
    
    # Handle extrapolation for any missing years at the beginning or end
    # Forward fill for early years
    energy_df['Population'] = energy_df['Population'].ffill()
    # Backward fill for later years
    energy_df['Population'] = energy_df['Population'].bfill()
    
    # Calculate total housing units
    energy_df['total_housing'] = energy_df[['heated_by_electricity', 'heated_by_gas', 'heated_by_fuel_oil', 
                                         'heated_by_other', 'no_heating', 'heated_by_lp_gas']].sum(axis=1)
    
    # Calculate housing units per capita ratio
    energy_df['housing_per_capita'] = energy_df['total_housing'] / energy_df['Population']
    
    # Replace any infinite values (from division by zero) with the median
    energy_df['housing_per_capita'] = energy_df['housing_per_capita'].replace([np.inf, -np.inf], np.nan)
    energy_df['housing_per_capita'] = energy_df['housing_per_capita'].fillna(energy_df['housing_per_capita'].median())
    
    # Calculate the percentage of each energy type
    for col in ['heated_by_electricity', 'heated_by_gas', 'heated_by_fuel_oil', 
                'heated_by_other', 'no_heating', 'heated_by_lp_gas']:
        energy_df[f'{col}_pct'] = energy_df[col] / energy_df['total_housing']
    
    # Fill potential NaN values with median
    for col in [f'{c}_pct' for c in ['heated_by_electricity', 'heated_by_gas', 'heated_by_fuel_oil', 
                                     'heated_by_other', 'no_heating', 'heated_by_lp_gas']]:
        energy_df[col] = energy_df[col].fillna(energy_df[col].median())
    
    # Calculate minimum values based on historical data to prevent unrealistically low values
    min_values = {}
    for col in ['heated_by_electricity', 'heated_by_gas', 'heated_by_fuel_oil', 
                'heated_by_other', 'no_heating', 'heated_by_lp_gas']:
        # Get non-zero values
        non_zero_values = energy_df[energy_df[col] > 0][col]
        if not non_zero_values.empty:
            # Set minimum to either 5% of the mean non-zero value or 1, whichever is larger
            min_values[col] = max(int(non_zero_values.mean() * 0.05), 1)
        else:
            min_values[col] = 1
    
    # Analyze population trends
    if len(energy_df) >= 2:
        # Calculate average annual growth rate over all available periods
        pop_df = energy_df[['Year', 'Population']].dropna()
        pop_df = pop_df.sort_values('Year')
        first_year = pop_df['Year'].min()
        last_year = pop_df['Year'].max()
        first_pop = pop_df.loc[pop_df['Year'] == first_year, 'Population'].iloc[0]
        last_pop = pop_df.loc[pop_df['Year'] == last_year, 'Population'].iloc[0]
        years_diff = last_year - first_year
        
        if years_diff > 0 and first_pop > 0:
            # Calculate compound annual growth rate
            annual_growth_rate = (last_pop / first_pop) ** (1 / years_diff) - 1
            
            # Ensure growth rate is realistic (between -0.5% and 3% per year)
            annual_growth_rate = max(-0.005, min(annual_growth_rate, 0.03))
            
            # Apply growth rate to project future population
            latest_pop = pop_df.loc[pop_df['Year'] == pop_df['Year'].max(), 'Population'].iloc[0]
            projected_populations = np.array([
                latest_pop * (1 + annual_growth_rate) ** (i + 1) for i in range(4)
            ])
        else:
            # Fallback to state average growth
            latest_pop = pop_df.loc[pop_df['Year'] == pop_df['Year'].max(), 'Population'].iloc[0]
            projected_populations = np.array([
                latest_pop * 1.05,  # 2025: 5% growth from latest
                latest_pop * 1.10,  # 2030: 10% growth
                latest_pop * 1.15,  # 2035: 15% growth
                latest_pop * 1.20   # 2040: 20% growth
            ])
    else:
        # Without enough population data, use state average growth rates
        last_known_year = energy_df['Year'].max()
        last_known_pop = energy_df.loc[energy_df['Year'] == last_known_year, 'Population'].iloc[0]
        
        # Estimate growth rates based on NC state average (approx 1% per year)
        projected_populations = np.array([
            last_known_pop * 1.05,  # 2025
            last_known_pop * 1.10,  # 2030
            last_known_pop * 1.15,  # 2035
            last_known_pop * 1.20   # 2040
        ])
    
    # Get most recent energy type distribution (percentages)
    latest_year = energy_df['Year'].max()
    latest_data = energy_df[energy_df['Year'] == latest_year]
    
    # Get most recent housing per capita ratio
    latest_housing_per_capita = latest_data['housing_per_capita'].iloc[0]
    
    # Create future dataframe with projected data
    future_data = []
    future_years = [2025, 2030, 2035, 2040]
    
    # Analyze trend for each energy type
    trends = {}
    
    for col in ['heated_by_electricity', 'heated_by_gas', 'heated_by_fuel_oil', 
                'heated_by_other', 'no_heating', 'heated_by_lp_gas']:
        pct_col = f'{col}_pct'
        
        # Check if there's a trend in the percentage over time
        if len(energy_df) >= 3:  # Need at least 3 years to detect a reliable trend
            # Calculate linear trend in percentage
            X_trend = energy_df['Year'].values.reshape(-1, 1)
            y_trend = energy_df[pct_col].values
            trend_model = LinearRegression()
            trend_model.fit(X_trend, y_trend)
            
            # Store trend
            trends[col] = {
                'slope': trend_model.coef_[0],
                'intercept': trend_model.intercept_
            }
        else:
            # Not enough data for trend, use latest percentage
            trends[col] = {
                'slope': 0,
                'intercept': latest_data[pct_col].iloc[0]
            }
    
    # Determine decline rate thresholds based on energy type
    decline_thresholds = {
        'heated_by_electricity': 0,  # No decline floor for electricity (growing)
        'heated_by_gas': 0.3,        # At least 30% of latest value for gas
        'heated_by_fuel_oil': 0.4,   # At least 40% of latest value for fuel oil
        'heated_by_other': 0.4,      # At least 40% of latest value for other
        'no_heating': 0.5,           # At least 50% of latest value for no heating
        'heated_by_lp_gas': 0.3      # At least 30% of latest value for LP gas
    }
    
    # Generate future projections
    for i, year in enumerate(future_years):
        future_row = {'Year': year, 'Population': projected_populations[i]}
        
        # Calculate expected total housing units based on population and housing per capita
        future_row['total_housing'] = future_row['Population'] * latest_housing_per_capita
        
        # First pass - calculate raw projections based on trends
        raw_projections = {}
        
        for col in ['heated_by_electricity', 'heated_by_gas', 'heated_by_fuel_oil', 
                    'heated_by_other', 'no_heating', 'heated_by_lp_gas']:
            # Predict percentage using trend
            trend = trends[col]
            predicted_pct = trend['slope'] * year + trend['intercept']
            
            # Ensure percentage is between 0 and 1
            predicted_pct = max(0, min(predicted_pct, 1))
            
            # Calculate predicted units
            predicted_units = int(round(future_row['total_housing'] * predicted_pct))
            
            # Get latest historical value for this energy type
            latest_value = latest_data[col].iloc[0]
            
            # Apply decline threshold - ensure prediction doesn't drop too much from latest value
            decline_threshold = decline_thresholds[col]
            min_value_by_threshold = int(latest_value * decline_threshold)
            
            # Apply minimum value constraint - use the larger of the threshold or the min from statistics
            final_min = max(min_value_by_threshold, min_values[col])
            
            # Ensure predicted value is at least the minimum
            if predicted_units < final_min:
                predicted_units = final_min
            
            # Store prediction
            raw_projections[col] = predicted_units
        
        # Second pass - normalize percentages to ensure they sum to total housing
        total_predicted = sum(raw_projections.values())
        
        if total_predicted > future_row['total_housing']:
            # Scale down proportionally if over total
            scale_factor = future_row['total_housing'] / total_predicted
            for col in raw_projections:
                raw_projections[col] = int(round(raw_projections[col] * scale_factor))
        
        # Add final projections to future row
        for col, value in raw_projections.items():
            future_row[col] = value
        
        future_data.append(future_row)
    
    # Create future DataFrame
    future_df = pd.DataFrame(future_data)
    
    # Get training data for Prophet - just the selected heating type
    df = pd.DataFrame({
        'Year': energy_df['Year'],
        heating_type: energy_df[heating_type],
        'Population': energy_df['Population']
    })
    
    # Now prepare for Prophet model
    prophet_df = pd.DataFrame({
        'ds': pd.to_datetime(df['Year'].astype(str)),
        'y': df[heating_type],
        'population': df['Population']
    })
    
    # Double check for NaN values in the prophet_df
    if prophet_df.isnull().any().any():
        print(f"Warning: Found NaN values in prophet_df for {county_name}, {heating_type}")
        # Fill NaN values in population column
        prophet_df['population'] = prophet_df['population'].interpolate().ffill().bfill()
        # Fill NaN values in y column with 0
        prophet_df['y'] = prophet_df['y'].fillna(0)
    
    # Create Prophet model
    model = Prophet(
        yearly_seasonality=False,
        growth='linear',
        changepoint_prior_scale=0.05  # More conservative to prevent overfit
    )
    
    # Add population as a regressor
    model.add_regressor('population')
    
    # Fit the model
    try:
        model.fit(prophet_df)
    except Exception as e:
        print(f"Error fitting Prophet model for {county_name}, {heating_type}: {e}")
        # Return distribution-based predictions as fallback
        predictions = pd.DataFrame({
            'County': county_name,
            'Year': future_df['Year'],
            heating_type: future_df[heating_type]
        })
        return predictions[['County', 'Year', heating_type]], None
    
    # Create future dataframe for Prophet
    prophet_future = pd.DataFrame({
        'ds': pd.to_datetime(future_df['Year'].astype(str)),
        'population': future_df['Population']
    })
    
    # Make prediction with Prophet
    forecast = model.predict(prophet_future)
    
    # Compare Prophet's predictions with our distribution-based predictions
    prophet_predictions = forecast['yhat'].values
    distribution_predictions = future_df[heating_type].values
    
    # Blend the predictions with different weights based on energy type
    if heating_type in ['heated_by_fuel_oil', 'heated_by_other']:
        # These types often show unrealistic decline in Prophet - use more distribution weight
        blend_weights = {
            2025: [0.3, 0.7],  # 30% Prophet, 70% distribution
            2030: [0.2, 0.8],  # 20% Prophet, 80% distribution
            2035: [0.15, 0.85], # 15% Prophet, 85% distribution
            2040: [0.1, 0.9]   # 10% Prophet, 90% distribution
        }
    elif heating_type in ['heated_by_lp_gas']:
        # Moderate blend for LP gas
        blend_weights = {
            2025: [0.4, 0.6],  # 40% Prophet, 60% distribution
            2030: [0.35, 0.65], # 35% Prophet, 65% distribution
            2035: [0.3, 0.7],  # 30% Prophet, 70% distribution
            2040: [0.25, 0.75]  # 25% Prophet, 75% distribution
        }
    else:
        # Standard blend for other types
        blend_weights = {
            2025: [0.5, 0.5],  # 50% Prophet, 50% distribution
            2030: [0.5, 0.5],  # 50% Prophet, 50% distribution
            2035: [0.5, 0.5],  # 50% Prophet, 50% distribution
            2040: [0.5, 0.5]   # 50% Prophet, 50% distribution
        }
    
    # Apply blending with year-specific weights
    blended_predictions = []
    for i, year in enumerate(future_years):
        weights = blend_weights[year]
        prophet_weight, dist_weight = weights
        
        # Apply weighted blend
        blended_value = int(round(
            prophet_weight * prophet_predictions[i] + dist_weight * distribution_predictions[i]
        ))
        
        # Ensure value is at least the minimum for this energy type
        blended_value = max(blended_value, min_values[heating_type])
        
        # For declining energy types, ensure we don't go below threshold of latest value
        if heating_type in ['heated_by_fuel_oil', 'heated_by_other', 'heated_by_lp_gas']:
            latest_value = latest_data[heating_type].iloc[0] 
            # Calculate minimum based on decline threshold and years into future
            years_out = (i + 1)
            # Apply progressively stronger floors for further years
            year_factor = max(0.1, 1 - (0.2 * years_out))  # 0.8, 0.6, 0.4, 0.2
            floor_value = int(latest_value * decline_thresholds[heating_type] * year_factor)
            
            # Apply the floor, but ensure it's at least min_values[heating_type]
            blended_value = max(blended_value, floor_value, min_values[heating_type])
        
        blended_predictions.append(blended_value)
    
    # Create prediction DataFrame for database
    predictions = pd.DataFrame({
        'County': county_name,
        'Year': future_df['Year'],
        heating_type: blended_predictions
    })
    
    print(f"Generated {len(predictions)} predictions for {county_name}, {heating_type}")
    print(f"  Prophet: {prophet_predictions.astype(int)}")
    print(f"  Distribution-based: {distribution_predictions}")
    print(f"  Blended (final): {blended_predictions}")
    
    # Also return confidence intervals for visualization
    prediction_intervals = pd.DataFrame({
        'County': county_name,
        'Year': future_df['Year'],
        f'{heating_type}_lower': np.maximum(forecast['yhat_lower'].values.astype(int), 
                                          [min_values[heating_type]] * len(future_years)),
        f'{heating_type}_upper': forecast['yhat_upper'].values.astype(int),
        heating_type: blended_predictions
    })
    
    return predictions[['County', 'Year', heating_type]], model, prediction_intervals

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = EnergyMapWindow()
    window.show()
    sys.exit(app.exec_())

