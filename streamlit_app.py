import streamlit as st
import pandas as pd
import plotly.express as px

# Set page title
st.title('Energy Consumption Dashboard')

# Add file uploader
uploaded_file = st.file_uploader("Upload your energy consumption data (CSV)", type="csv")

if uploaded_file is not None:
    # Read the data
    data = pd.read_csv(uploaded_file, parse_dates={'datetime': ['Date', 'Time']}, 
                       sep=';', infer_datetime_format=True)
    data.set_index('datetime', inplace=True)

    # Display interactive time series plot
    fig = px.line(data, y='Global_active_power',
                  title='Energy Consumption Over Time')
    st.plotly_chart(fig)

    # Show peak hours analysis
    st.subheader('Peak Hours Analysis')
    peak_hours = data.groupby(data.index.hour)['Global_active_power'].mean().sort_values(ascending=False).head()
    st.bar_chart(peak_hours)

    # Show optimization suggestions
    st.subheader('Optimization Suggestions')
    def suggest_shifting(hour):
        if hour in peak_hours.index:
            return "Shift to off-peak hours"
        return "No change needed"
    
    data['optimization_suggestion'] = data.index.hour.map(suggest_shifting)
    st.write(data[['Global_active_power', 'optimization_suggestion']].head(10))

else:
    st.write("Please upload a CSV file to view the dashboard")