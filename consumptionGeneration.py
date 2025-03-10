import sqlite3
import pandas as pd
from prophet import Prophet

def setup_database():
    """Setup and verify the energy database"""
    try:
        # Connect to database
        conn = sqlite3.connect('nc_energy.db')
        print("Connected to database")
        
        # Check if energy_predictions table exists and has data
        check_query = """
        SELECT COUNT(*) FROM sqlite_master 
        WHERE type='table' AND name='energy_predictions';
        """
        if pd.read_sql_query(check_query, conn).iloc[0,0] == 0:
            print("Generating predictions table...")
            # Get historical data
            historical = pd.read_sql_query("""
                SELECT * FROM energy_consumption
                ORDER BY County, Year
            """, conn)
            
            # Create predictions table
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS energy_predictions (
                County TEXT,
                Year INTEGER,
                heated_by_electricity INTEGER,
                heated_by_gas INTEGER,
                heated_by_fuel_oil INTEGER,
                heated_by_other INTEGER,
                no_heating INTEGER,
                heated_by_lp_gas INTEGER,
                PRIMARY KEY (County, Year)
            );
            """
            conn.execute(create_table_sql)
            
            # Load predictions from combined data if available
            try:
                combined_data = pd.read_csv('nc_energy_combined.csv')
                predictions = combined_data[combined_data['Year'] >= 2025]
                predictions.to_sql('energy_predictions', conn, if_exists='replace', index=False)
                print("Loaded predictions from combined data")
            except Exception as e:
                print(f"Error loading predictions: {e}")
                return False
        
        print("Database setup complete")
        return True
        
    except Exception as e:
        print(f"Database setup error: {e}")
        return False

if __name__ == "__main__":
    setup_database()