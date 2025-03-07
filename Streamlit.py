import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor  # Import DecisionTreeRegressor
from sklearn.svm import SVR  # Import SVR
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# Add custom CSS to style buttons
st.markdown(
    """
    <style>
    .stButton>button {
        background-color: lightyellow;
        color: black;
        border: None;
        border-radius: 5px;
        padding: 10px;
        font-size: 16px;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: yellow;
    }
    </style>
    """,
    unsafe_allow_html=True
)

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

    # Button 1: Total Ad Spend by State
    if st.button("Total Ad Spend by State"):
        state_ad_spend = merged_data.groupby('State')['Amount spent (INR)'].sum().reset_index()
        fig = go.Figure(data=[go.Bar(x=state_ad_spend['State'], y=state_ad_spend['Amount spent (INR)'])])
        fig.update_layout(title='Total Ad Spend by State', xaxis_title='State', yaxis_title='Amount Spent (INR)')
        st.plotly_chart(fig)

    # Button 2: Average Voter Turnout by State
    if st.button("Average Voter Turnout by State"):
        state_voter_turnout = merged_data.groupby('State')['Polled (%)'].mean().reset_index()
        fig = px.bar(state_voter_turnout, x='State', y='Polled (%)', color='Polled (%)',
                     title='Average Voter Turnout by State')
        fig.update_layout(xaxis_tickangle=-90, width=1000, height=600)
        st.plotly_chart(fig)

    # Button 3: Top 5 Parties by Ad Spend
    if st.button("Top 5 Parties by Ad Spend"):
        advertisers['Amount spent (INR)'] = pd.to_numeric(advertisers['Amount spent (INR)'], errors='coerce')
        top_5_parties = advertisers.groupby('Page name')['Amount spent (INR)'].sum().sort_values(ascending=False).head(5)
        fig = px.pie(top_5_parties, values='Amount spent (INR)', names=top_5_parties.index, hole=0.4,
                     title='Top 5 Parties by Ad Spend')
        st.plotly_chart(fig)

    # Button 4: Ad Spend vs Voter Turnout
    if st.button("Ad Spend vs Voter Turnout"):
        fig = px.scatter(merged_data, x='Amount spent (INR)', y='Polled (%)', color='State',
                         title='Ad Spend vs Voter Turnout by Constituency')
        st.plotly_chart(fig)

    # Button 5: Distribution of Ad Spend
    if st.button("Distribution of Ad Spend"):
        fig = px.histogram(merged_data, x='Amount spent (INR)', nbins=30, marginal='box',
                           title='Distribution of Ad Spend')
        st.plotly_chart(fig)

    # Button 6: Ad Spend and Voter Turnout by Election Phase
    if st.button("Ad Spend and Voter Turnout by Election Phase"):
        phase_analysis = merged_data.groupby('Phase').agg({'Amount spent (INR)': 'sum', 'Polled (%)': 'mean'}).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=phase_analysis['Phase'], y=phase_analysis['Amount spent (INR)'], name='Ad Spend'))
        fig.add_trace(go.Scatter(x=phase_analysis['Phase'], y=phase_analysis['Polled (%)'], name='Voter Turnout', yaxis='y2'))
        fig.update_layout(title='Ad Spend and Voter Turnout by Phase', yaxis2=dict(overlaying='y', side='right'))
        st.plotly_chart(fig)

    # Button 7: Comparison of Actual and Predicted Spending
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

    # Button 8: Spending by Platform
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

    # Button 9: Political Party Spending by Platform
    if st.button("Political Party Spending by Platform"):
        # Example data with platform and party spending
        data = {
            'Page_Name': ['Bharatiya Janata Party (BJP)', 'Indian National Congress', 'Aam Aadmi Party',
                          'Bharatiya Janata Party (BJP)', 'Indian National Congress', 'Aam Aadmi Party'],
            'Platform': ['Facebook', 'Instagram', 'Facebook', 'Instagram', 'Facebook', 'Instagram'],
            'Amount_Spent_INR': [193854342, 108787100, 50000000, 120000000, 60000000, 30000000]
        }
        df_spending = pd.DataFrame(data)

        # Create a bar chart using Plotly Express
        fig = px.bar(df_spending, x='Page_Name', y='Amount_Spent_INR', color='Platform', barmode='group',
                     title='Political Party Spending by Platform', text_auto=True)

        # Display the chart
        st.plotly_chart(fig)

    # Button 10: Election Spending Predictions using Multiple Models
        # Button 10: Election Spending Predictions using Multiple Models
    if st.button("Election Spending Predictions using Multiple Models"):
        # Example data (replace with actual data if available)
        data = {
            'Year': [2014, 2019, 2024],  # Years of elections
            'BJP_Amount_Spent_INR': [120000000, 193854342, 300000000],  # BJP spendings
            'Congress_Amount_Spent_INR': [80000000, 108787100, 150000000],  # Congress spendings
            'AAP_Amount_Spent_INR': [20000000, 50000000, 80000000]  # AAP spendings
        }

        df = pd.DataFrame(data)

        # Convert Year into numerical form for regression
        df['Year_Num'] = df['Year'] - df['Year'].min()

        # Define models for prediction
        models = {
            'Linear Regression': LinearRegression(),
            'Polynomial Regression (degree=2)': None,  # Handled separately
            'Decision Tree': DecisionTreeRegressor(),
            'SVR': SVR(kernel='rbf')
        }

        # Helper function to train models, predict, and calculate accuracy for each party
        def predict_spending(party_column):
            X = df[['Year_Num']]  # Independent variable: Year
            y = df[party_column]  # Dependent variable: Amount Spent

            predictions = {}
            accuracies = {}

            # Linear Regression
            model_lr = models['Linear Regression']
            model_lr.fit(X, y)
            predictions['Linear Regression'] = model_lr.predict([[0], [5], [10], [15]])
            accuracies['Linear Regression'] = mean_absolute_error(y, model_lr.predict(X))

            # Polynomial Regression (degree=2)
            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X)
            model_poly = LinearRegression()
            model_poly.fit(X_poly, y)
            predictions['Polynomial Regression (degree=2)'] = model_poly.predict(poly.transform([[0], [5], [10], [15]]))
            accuracies['Polynomial Regression (degree=2)'] = mean_absolute_error(y, model_poly.predict(X_poly))

            # Decision Tree
            model_dt = models['Decision Tree']
            model_dt.fit(X, y)
            predictions['Decision Tree'] = model_dt.predict([[0], [5], [10], [15]])
            accuracies['Decision Tree'] = mean_absolute_error(y, model_dt.predict(X))

            # SVR
            model_svr = models['SVR']
            model_svr.fit(X, y)
            predictions['SVR'] = model_svr.predict([[0], [5], [10], [15]])
            accuracies['SVR'] = mean_absolute_error(y, model_svr.predict(X))

            return predictions, accuracies

        # Predict spending for BJP, Congress, and AAP
        bjp_predictions, bjp_accuracies = predict_spending('BJP_Amount_Spent_INR')
        congress_predictions, congress_accuracies = predict_spending('Congress_Amount_Spent_INR')
        aap_predictions, aap_accuracies = predict_spending('AAP_Amount_Spent_INR')

        # Function to plot predictions for a specific party
        def plot_predictions(party_name, actual_values, predictions, accuracies):
            plt.figure(figsize=(10, 6))
            years = np.array([2014, 2019, 2024, 2029])

            # Actual values
            plt.plot(years[:3], actual_values, label=f'Actual {party_name} Spending', marker='o', color='black', linestyle='--')

            # Plot predicted spending
            for model_name, preds in predictions.items():
                plt.plot(years, preds, label=f'{party_name} {model_name}', linestyle='-', marker='o')

            # Formatting the plot
            plt.title(f'{party_name} Election Ad Spending Predictions for 2029')
            plt.xlabel('Year')
            plt.ylabel('Amount Spent (INR)')
            plt.xticks(years)
            plt.legend()
            plt.grid(True)

            # Show the plot
            st.pyplot(plt)

            # Print accuracy scores for the party
            st.write(f"Accuracy Scores (Mean Absolute Error) for {party_name}:")
            for model_name, score in accuracies.items():
                st.write(f"{model_name}: {score:.2f} INR")

        # Plot predictions for each party
        plot_predictions('BJP', df['BJP_Amount_Spent_INR'].values, bjp_predictions, bjp_accuracies)
        plot_predictions('Congress', df['Congress_Amount_Spent_INR'].values, congress_predictions, congress_accuracies)
        plot_predictions('AAP', df['AAP_Amount_Spent_INR'].values, aap_predictions, aap_accuracies)
