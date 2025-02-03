import pandas as pd

# Load CSV with tab delimiter
df = pd.read_csv('energy-and-utilities-linc.csv', sep='\t')

# Save with semicolon delimiter
df.to_csv('energy-and-utilities-linc-converted.csv', sep=';', index=False)

