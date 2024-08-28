import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from geopy.distance import great_circle
import numpy as np

# Function to load data
@st.cache_data
def load_data():
    file_path = 'https://raw.githubusercontent.com/Lovelylove03/Gourmet-Restaurant/main/df_mich.csv'  
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
def recommend_restaurants(data, cuisine_preference, price_range, location_preference, top_n=5):
    # Filter restaurants based on user preferences
    filtered_data = data[
        (data['Cuisine'].str.contains(cuisine_preference, case=False)) &
        (data['Price'].str.contains(price_range, case=False)) &
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

# Function to filter restaurants by proximity to a given location
def filter_by_location(data, user_lat, user_lon, radius_km=10):
    def is_within_radius(row):
        return great_circle((user_lat, user_lon), (row['Latitude'], row['Longitude'])).kilometers <= radius_km
    
    return data[data.apply(is_within_radius, axis=1)]

# Function to format price
def format_price(price):
    # Assume the price column contains numerical values with currency symbols
    return f"{price} EUR"  # Adjust according to the actual currency and format

# Streamlit App
def main():
    st.title("Gourmet Restaurant Recommendation System")
    
    # Load data
    data = load_data()
    data = preprocess_data(data)
    
    # User Inputs
    st.sidebar.header('Customize Your Search')
    cuisine_preference = st.sidebar.selectbox("Choose Cuisine Type", data['Cuisine'].unique())
    price_range = st.sidebar.selectbox("Choose Price Range", data['Price'].unique())
    location_preference = st.sidebar.selectbox("Choose Location", data['Location'].unique())
    
    # User location input for proximity filter
    user_location = st.sidebar.text_input("Enter Your Town (e.g., Marseille, France)")
    
    if user_location:
        # Assume that you have a function to get latitude and longitude from user location
        user_lat, user_lon = get_lat_lon_from_location(user_location)  # Implement this function
        data = filter_by_location(data, user_lat, user_lon)
    
    # Recommend Restaurants
    if st.sidebar.button("Get Recommendations"):
        recommendations = recommend_restaurants(data, cuisine_preference, price_range, location_preference, top_n=5)
        if not recommendations.empty:
            for i, row in recommendations.iterrows():
                st.subheader(row['Name'])
                st.write(f"Cuisine: {row['Cuisine']}")
                st.write(f"Price: {format_price(row['Price'])}")
                st.write(f"Location: {row['Location']}")
                st.write(f"Award: {row['Award']}")
                st.write(f"Phone: {row.get('PhoneNumber', 'N/A')}")
                st.write(f"Description: {row['Description']}")
                st.write(f"Facilities and Services: {row['FacilitiesAndServices']}")
                st.map(pd.DataFrame([[row['Latitude'], row['Longitude']]], columns=['lat', 'lon']))

def get_lat_lon_from_location(location):
    # Placeholder for actual geocoding implementation
    # Example: Use a geocoding API to get latitude and longitude from location string
    return 43.2965, 5.3698  # Example coordinates for Marseille, France

if __name__ == '__main__':
    main()
