import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Set page title
st.title('Analysis on Election Ad Spending')

# Load the CSV files (use your own file paths)
st.sidebar.title('Navigation')
uploaded_files = st.sidebar.file_uploader("Upload your CSV files", accept_multiple_files=True)

# Store CSV data in dataframes
results = None
advertisers = None
locations = None

for uploaded_file in uploaded_files:
    if 'results' in uploaded_file.name:
        results = pd.read_csv(uploaded_file)
    elif 'advertisers' in uploaded_file.name:
        advertisers = pd.read_csv(uploaded_file)
    elif 'locations' in uploaded_file.name:
        locations = pd.read_csv(uploaded_file)

# If no CSVs are uploaded, display error
if results is None or advertisers is None or locations is None:
    st.warning('Please upload results, advertisers, and locations CSV files')
else:
    # Display the CSVs
    if st.checkbox('Show Results Data'):
        st.write(results)
    if st.checkbox('Show Advertisers Data'):
        st.write(advertisers)
    if st.checkbox('Show Locations Data'):
        st.write(locations)

    # Merge datasets based on a key column ('State' or 'Location name')
    merged_data = results.merge(locations, left_on='State', right_on='Location name', how='left')

    # Graph Buttons
    st.subheader('Election Ad Spending Analysis')

    # Existing button for total ad spend by state
    if st.button("Total Ad Spend by State"):
        state_ad_spend = merged_data.groupby('State')['Amount spent (INR)'].sum().reset_index()
        fig = go.Figure(data=[go.Bar(x=state_ad_spend['State'], y=state_ad_spend['Amount spent (INR)'])])
        fig.update_layout(title='Total Ad Spend by State', xaxis_title='State', yaxis_title='Amount Spent (INR)')
        st.plotly_chart(fig)

    # Existing button for average voter turnout by state
    if st.button("Average Voter Turnout by State"):
        state_voter_turnout = merged_data.groupby('State')['Polled (%)'].mean().reset_index()
        fig = px.bar(state_voter_turnout, x='State', y='Polled (%)', color='Polled (%)',
                     title='Average Voter Turnout by State')
        fig.update_layout(xaxis_tickangle=-90, width=1000, height=600)
        st.plotly_chart(fig)

    # Existing button for top 5 parties by ad spend
    if st.button("Top 5 Parties by Ad Spend"):
        advertisers['Amount spent (INR)'] = pd.to_numeric(advertisers['Amount spent (INR)'], errors='coerce')
        top_5_parties = advertisers.groupby('Page name')['Amount spent (INR)'].sum().sort_values(ascending=False).head(5)
        fig = px.pie(top_5_parties, values='Amount spent (INR)', names=top_5_parties.index, hole=0.4,
                     title='Top 5 Parties by Ad Spend')
        st.plotly_chart(fig)

    # Existing button for ad spend vs voter turnout
    if st.button("Ad Spend vs Voter Turnout"):
        fig = px.scatter(merged_data, x='Amount spent (INR)', y='Polled (%)', color='State',
                         title='Ad Spend vs Voter Turnout by Constituency')
        st.plotly_chart(fig)

    # Existing button for distribution of ad spend
    if st.button("Distribution of Ad Spend"):
        fig = px.histogram(merged_data, x='Amount spent (INR)', nbins=30, marginal='box',
                           title='Distribution of Ad Spend')
        st.plotly_chart(fig)

    # Existing button for ad spend and voter turnout by election phase
    if st.button("Ad Spend and Voter Turnout by Election Phase"):
        phase_analysis = merged_data.groupby('Phase').agg({'Amount spent (INR)': 'sum', 'Polled (%)': 'mean'}).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=phase_analysis['Phase'], y=phase_analysis['Amount spent (INR)'], name='Ad Spend'))
        fig.add_trace(go.Scatter(x=phase_analysis['Phase'], y=phase_analysis['Polled (%)'], name='Voter Turnout', yaxis='y2'))
        fig.update_layout(title='Ad Spend and Voter Turnout by Phase', yaxis2=dict(overlaying='y', side='right'))
        st.plotly_chart(fig)

    # Existing button for comparison of actual and predicted spending
    if st.button("Comparison of Actual and Predicted Spending"):
        # Data loading with actual columns names
        data = {
            'Page_Name': ['Bharatiya Janata Party (BJP)', 'Indian National Congress', 'Aam Aadmi Party'],
            'Number_of_Ads_in_Library': [43455, 846, 1200],
            'Amount_Spent_INR': [193854342, 108787100, 50000000]
        }
        df = pd.DataFrame(data)

        # Split data into features and target
        X = df[['Number_of_Ads_in_Library']]
        y = df['Amount_Spent_INR']

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create a RandomForest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Plotting results
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual Amount Spent', s=100, marker='o')  # Actual values
        plt.scatter(range(len(predictions)), predictions, color='red', label='Predicted Amount Spent', s=100, marker='^')  # Predictions
        plt.xticks(range(len(y_test)), y_test.index, rotation=45)
        plt.xlabel('Index')
        plt.ylabel('Amount Spent (INR)')
        plt.title('Comparison of Actual and Predicted Spending')
        plt.legend()
        plt.grid(True)

        # Show the plot
        st.pyplot(plt)

        # Calculate and print RMSE
        rmse = mean_squared_error(y_test, predictions, squared=False)
        st.write(f'Root Mean Square Error: {rmse}')

    # Existing button for spending by platform
    if st.button("Spending by Platform"):
        # Load dataset into a DataFrame (Replace this with your actual dataset)
        data = {
            'Page_Name': ['Bharatiya Janata Party (BJP)', 'Indian National Congress', 'Aam Aadmi Party',
                          'Bharatiya Janata Party (BJP)', 'Indian National Congress', 'Aam Aadmi Party'],
            'Platform': ['Facebook', 'Instagram', 'Facebook', 'Facebook', 'Instagram', 'Instagram'],
            'Amount_Spent_INR': [193854342, 108787100, 50000000, 120000000, 60000000, 30000000]
        }
        
        df = pd.DataFrame(data)

        # Group by 'Platform' and sum the 'Amount_Spent_INR'
        spending_by_platform = df.groupby('Platform')['Amount_Spent_INR'].sum()

        # Sort the result in descending order
        spending_by_platform_sorted = spending_by_platform.sort_values(ascending=False)

        # Print the results to Streamlit
        st.write("Total Spending on Each Platform:")
        st.write(spending_by_platform_sorted)

        # Plot the result as a bar chart
        plt.figure(figsize=(6, 4))
        spending_by_platform_sorted.plot(kind='bar', color=['blue', 'orange'], title='Total Spending by Platform', ylabel='Amount Spent (INR)', xlabel='Platform')
        plt.xticks(rotation=0)
        plt.tight_layout()

        # Show the plot in Streamlit
        st.pyplot(plt)

    # New button for political party spending by platform
    if st.button("Political Party Spending by Platform"):
        # Example data with platform and party spending
        data = {
            'Page_Name': ['Bharatiya Janata Party (BJP)', 'Indian National Congress', 'Aam Aadmi Party',
                          'Bharatiya Janata Party (BJP)', 'Indian National Congress', 'Aam Aadmi Party'],
            'Platform': ['Facebook', 'Instagram', 'Facebook', 'Instagram', 'Facebook', 'Instagram'],
            'Amount_Spent_INR': [193854342, 108787100, 50000000, 120000000, 60000000, 30000000]
        }

        # Create DataFrame
        df_spending = pd.DataFrame(data)

        # Create a pivot table to show spending by party and platform
        pivot_table = df_spending.pivot_table(values='Amount_Spent_INR', index='Page_Name', columns='Platform', aggfunc='sum', fill_value=0)

        # Plot settings for an attractive look
        sns.set(style="whitegrid")  # Set grid style
        colors = ['#1f77b4', '#ff7f0e']  # Custom color palette for Facebook and Instagram

        # Plot the result as a grouped bar chart
        ax = pivot_table.plot(kind='bar', figsize=(12, 7), color=colors, edgecolor='black', linewidth=1)

        # Title and labels with larger font size for better readability
        plt.title('Political Party Spending by Platform (Facebook vs Instagram)', fontsize=16, weight='bold')
        plt.ylabel('Amount Spent (INR)', fontsize=14)
        plt.xlabel('Political Party', fontsize=14)

        # Adding value labels on top of each bar
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height()):,}', (p.get_x() * 1.005, p.get_height() * 1.01), fontsize=12)

        # Rotate x-axis labels for better visibility
        plt.xticks(rotation=45, fontsize=12, weight='bold')

        # Adding gridlines for y-axis
        plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.7)

        # Customizing the legend
        plt.legend(title="Platform", fontsize=12, title_fontsize=14)

        # Show the plot
        plt.tight_layout()  # Adjust layout for better appearance
        st.pyplot(plt)

    # New button for election spending predictions
    if st.button("Predict Election Spending for 2029"):
        # Example data (assuming previous elections or some historical data)
        data = {
            'Year': [2014, 2019, 2024],  # Years of elections (replace with actual data)
            'BJP_Amount_Spent_INR': [120000000, 193854342, 300000000],  # Example spendings (replace with actual data)
            'Congress_Amount_Spent_INR': [80000000, 108787100, 150000000],
            'AAP_Amount_Spent_INR': [20000000, 50000000, 80000000]
        }

        df = pd.DataFrame(data)

        # Convert Year into numerical form for regression
        df['Year_Num'] = df['Year'] - df['Year'].min()

        # Set up the linear regression model for each party
        def predict_spending(party_column):
            X = df[['Year_Num']]  # Independent variable: year
            y = df[party_column]  # Dependent variable: amount spent

            model = LinearRegression()
            model.fit(X, y)

            # Predict next election spending (assuming next election is in 2029)
            next_year = 2029 - df['Year'].min()
            prediction = model.predict([[next_year]])

            return prediction[0], model

        # Predict spending for BJP, Congress, and AAP
        bjp_spending, bjp_model = predict_spending('BJP_Amount_Spent_INR')
        congress_spending, congress_model = predict_spending('Congress_Amount_Spent_INR')
        aap_spending, aap_model = predict_spending('AAP_Amount_Spent_INR')

        # Show predictions
        st.write(f"BJP predicted spending for 2029: {bjp_spending:.2f} INR")
        st.write(f"Congress predicted spending for 2029: {congress_spending:.2f} INR")
        st.write(f"AAP predicted spending for 2029: {aap_spending:.2f} INR")

        # Plot the predictions
        plt.figure(figsize=(10, 6))
        years = np.array([2014, 2019, 2024, 2029])

        # BJP Prediction
        bjp_predicted = bjp_model.predict(np.array([[0], [5], [10], [15]]))
        plt.plot(years, bjp_predicted, label='BJP', marker='o')

        # Congress Prediction
        congress_predicted = congress_model.predict(np.array([[0], [5], [10], [15]]))
        plt.plot(years, congress_predicted, label='Congress', marker='o')

        # AAP Prediction
        aap_predicted = aap_model.predict(np.array([[0], [5], [10], [15]]))
        plt.plot(years, aap_predicted, label='AAP', marker='o')

        plt.title('Election Spending Predictions (2029)')
        plt.xlabel('Year')
        plt.ylabel('Amount Spent (INR)')
        plt.legend()
        plt.grid(True)
        plt.xticks(years)
        st.pyplot(plt)
