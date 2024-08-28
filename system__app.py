import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import requests

# API endpoint and key for currency conversion (example using exchangerate-api.com)
API_URL = 'https://v6.exchangerate-api.com/v6/YOUR_API_KEY/latest/USD'

# Function to get currency conversion rates
def get_conversion_rates():
    response = requests.get(API_URL)
    data = response.json()
    return data['conversion_rates']

# Function to convert price to selected currency
def convert_price(price, from_currency, to_currency, rates):
    if from_currency == to_currency:
        return price
    rate = rates[to_currency] / rates[from_currency]
    return price * rate

# Function to load data
@st.cache_data
def load_data():
    file_path = 'https://raw.githubusercontent.com/Lovelylove03/Gourmet-Restaurant/main/datamission.csv'  
    data = pd.read_csv(file_path)
    return data

# Function to preprocess data for similarity calculation
def preprocess_data(data):
    # Handling missing values
    data['PhoneNumber'].fillna('Not Available', inplace=True)
    data['WebsiteUrl'].fillna('No Website', inplace=True)
    data['FacilitiesAndServices'].fillna('None', inplace=True)
    data['Description'].fillna('No Description', inplace=True)

    # Create a combined feature for similarity calculation
    data['Combined'] = data['Cuisine'] + ' ' + data['Price'] + ' ' + data['Location']
    return data

# Function to recommend restaurants based on user preferences
def recommend_restaurants(data, cuisine_preference, min_price, max_price, location_preference, top_n=5):
    # Convert price range to numerical values
    data['PriceNumeric'] = data['Price'].replace('[\$,]', '', regex=True).astype(float)
    
    # Filter restaurants based on user preferences
    filtered_data = data[
        (data['Cuisine'].str.contains(cuisine_preference, case=False)) &
        (data['PriceNumeric'] >= min_price) &
        (data['PriceNumeric'] <= max_price) &
        (data['Location'].str.contains(location_preference, case=False))
    ]
    
    if filtered_data.empty:
        st.write("No restaurants found matching your preferences.")
        return pd.DataFrame()  # Return empty DataFrame
    
    # Use TF-IDF Vectorizer to compute similarity based on the combined feature
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(filtered_data['Combined'])
    
    # Compute cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Get restaurant indices
    indices = pd.Series(filtered_data.index, index=filtered_data['Name']).drop_duplicates()

    # Recommendation function
    def get_recommendations(name, cosine_sim=cosine_sim):
        idx = indices[name]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n + 1]
        restaurant_indices = [i[0] for i in sim_scores]
        return filtered_data.iloc[restaurant_indices]
    
    # Provide recommendations for the first restaurant in the filtered list
    recommended_restaurants = get_recommendations(filtered_data.iloc[0]['Name'])
    
    return recommended_restaurants

# Function to format price based on selected currency
def format_price(price, currency, rates):
    return f"{price:.2f} {currency}"

# Function to get latitude and longitude from location (Placeholder function)
def get_lat_lon_from_location(location):
    # Placeholder for actual geocoding implementation
    # Example: Use a geocoding API to get latitude and longitude from location string
    return 43.2965, 5.3698  # Example coordinates for Marseille, France

# Streamlit App
def main():
    st.title("Gourmet Restaurant Recommendation System")
    
    # Load data
    data = load_data()
    data = preprocess_data(data)
    
    # Get conversion rates
    rates = get_conversion_rates()
    
    # User Inputs
    st.sidebar.header('Customize Your Search')
    cuisine_preference = st.sidebar.selectbox("Choose Cuisine Type", data['Cuisine'].unique())
    currency = st.sidebar.selectbox("Choose Currency", list(rates.keys()))
    
    # User inputs for numerical price range
    min_price = st.sidebar.number_input("Min Price", min_value=0.0, value=0.0)
    max_price = st.sidebar.number_input("Max Price", min_value=0.0, value=100.0)
    
    location_preference = st.sidebar.selectbox("Choose Location", data['Location'].unique())
    
    # User location input for proximity filter
    user_location = st.sidebar.text_input("Enter Your Location (e.g., Marseille, France)")
    
    if user_location:
        user_lat, user_lon = get_lat_lon_from_location(user_location)  # Implement this function
        data = filter_by_location(data, user_lat, user_lon)
    
    # Recommend Restaurants
    if st.sidebar.button("Get Recommendations"):
        recommendations = recommend_restaurants(data, cuisine_preference, min_price, max_price, location_preference, top_n=5)
        if not recommendations.empty:
            for i, row in recommendations.iterrows():
                price_numeric = float(row['Price'].replace('$', '').replace(',', '').strip())
                price_converted = convert_price(price_numeric, 'USD', currency, rates)  # Assuming 'USD' as base currency
                st.subheader(row['Name'])
                st.write(f"Cuisine: {row['Cuisine']}")
                st.write(f"Price: {format_price(price_converted, currency, rates)}")
                st.write(f"Location: {row['Location']}")
                st.write(f"Award: {row['Award']}")
                st.write(f"Phone: {row.get('PhoneNumber', 'N/A')}")
                st.write(f"Description: {row['Description']}")
                st.write(f"Facilities and Services: {row['FacilitiesAndServices']}")
                st.map(pd.DataFrame([[row['Latitude'], row['Longitude']]], columns=['lat', 'lon']))

if __name__ == '__main__':
    main()
