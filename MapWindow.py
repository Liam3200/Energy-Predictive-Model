import sys
import os
import json
import pandas as pd
import geopandas as gpd
import folium
from folium import Choropleth, LayerControl, features
from branca.colormap import linear
from PyQt5.QtWidgets import *
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl, Qt
import sqlite3
import urllib.request

class EnergyMapWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('NC Energy Consumption Map')
        self.setGeometry(100, 100, 1200, 800)
        self.init_ui()
        self.load_data()

    def init_ui(self):
        """Initialize the user interface components"""
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setSpacing(10)  # Add spacing between layout elements
        
        # Controls area - make it more compact
        controls_widget = QWidget()
        controls_widget.setMaximumHeight(100)  # Limit height of controls
        controls_layout = QHBoxLayout(controls_widget)
        controls_layout.setContentsMargins(10, 5, 10, 5)  # Reduce margins
        
        # Year slider with value label - Add tick interval
        year_widget = QWidget()
        year_layout = QVBoxLayout(year_widget)
        year_layout.setSpacing(5)
        
        year_label = QLabel("Year:")
        year_label.setStyleSheet("font-weight: bold;")
        year_layout.addWidget(year_label)
        
        self.year_slider = QSlider(Qt.Horizontal)
        self.year_slider.setMinimumWidth(200)  # Set minimum width
        self.year_slider.setTickPosition(QSlider.TicksBelow)  # Show tick marks
        self.year_slider.setTickInterval(10)  # Set tick interval to 10 years
        self.year_slider.setSingleStep(10)  # Set step size to 10 years
        self.year_slider.setPageStep(10)     # Set page step to 10 years
        self.year_value_label = QLabel()     # Add label to show current year
        self.year_slider.valueChanged.connect(self.update_year_label)
        year_layout.addWidget(self.year_slider)
        year_layout.addWidget(self.year_value_label)
        controls_layout.addWidget(year_widget)
        
        # Add vertical line separator
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setStyleSheet("color: #cccccc;")
        controls_layout.addWidget(line)
        
        # Data source selection
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
        
        # Add controls to main layout
        layout.addWidget(controls_widget)
        
        # Map view
        self.web_view = QWebEngineView()
        self.web_view.setMinimumSize(800, 600)
        self.web_view.setStyleSheet("border: 1px solid #cccccc; border-radius: 4px;")
        layout.addWidget(self.web_view)
        
        # Connect signals
        self.year_slider.valueChanged.connect(self.update_map)
        self.heating_dropdown.currentIndexChanged.connect(self.update_map)
        self.source_dropdown.currentIndexChanged.connect(self.update_map)

    def update_year_label(self):
        """Update the year label when slider changes"""
        self.year_value_label.setText(str(self.year_slider.value()))

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
                min_year = int(self.all_data['Year'].min())
                max_year = int(self.all_data['Year'].max())
                print(f"Year range: {min_year}-{max_year}")
                
                # Set year slider range
                self.year_slider.setMinimum(min_year)
                self.year_slider.setMaximum(max_year)
                self.year_slider.setValue(min_year)
                self.update_year_label()
                
                print(f"Data loaded successfully. Year range: {min_year}-{max_year}")
                
                # Update map initially
                if heating_types:
                    self.heating_dropdown.setCurrentIndex(0)
                    self.update_map()
            else:
                print("Warning: No data available in database")
                QMessageBox.warning(self, "Warning", "No data available in database")
            
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
            year = self.year_slider.value()
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
                # Temporarily block signals to avoid recursive updates
                self.year_slider.blockSignals(True)
                self.year_slider.setValue(year)
                self.year_slider.blockSignals(False)
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
            
            # Extract selected heating type data
            df_subset = df[['County', heating_type]].copy()
            df_subset = df_subset.rename(columns={heating_type: 'value'})
            print(f"Value range: {df_subset['value'].min()} to {df_subset['value'].max()}")
            
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
                            "value": float(value)
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
                
                # Add tooltips
                folium.GeoJsonTooltip(
                    fields=['NAME', 'value'],
                    aliases=['County:', f'{heating_type}:'],
                    style='background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;',
                    localize=True,
                    sticky=False,
                    labels=True,
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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = EnergyMapWindow()
    window.show()
    sys.exit(app.exec_())

