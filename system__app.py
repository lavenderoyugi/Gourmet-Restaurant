import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Function to load data
@st.cache_data
def load_data():
    file_path = 'https://raw.githubusercontent.com/Lovelylove03/Gourmet-Restaurant/main/datamission.csv'  
    data = pd.read_csv(file_path)


# Function to preprocess data for similarity calculation
def preprocess_data(data):
# 1. Handling `PhoneNumber` (3.01% missing)
# Impute missing values with 'Not Available' or a placeholder for phone numbers
     data['PhoneNumber'].fillna('Not Available', inplace=True)

# 2. Handling `WebsiteUrl` (16.95% missing)
# Impute missing values with 'No Website' or another appropriate value
data['WebsiteUrl'].fillna('No Website', inplace=True)

# 3. Handling `FacilitiesAndServices` (5.27% missing)
# You can fill with 'None' if the missing value implies no facilities or services listed
data['FacilitiesAndServices'].fillna('None', inplace=True)

# 4. Handling `Description` (0.02% missing)
# Since it's a very small percentage, you could drop the rows or fill with 'No Description'
data['Description'].fillna('No Description', inplace=True)

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
    
    # Recommend Restaurants
    if st.sidebar.button("Get Recommendations"):
        recommendations = recommend_restaurants(data, cuisine_preference, price_range, location_preference, top_n=5)
        if not recommendations.empty:
            for i, row in recommendations.iterrows():
                st.subheader(row['Name'])
                # Placeholder for images
                st.image("https://docs.developer.yelp.com/docs/fusion-intro")
                st.image("https://www.google.com/maps/search/Restaurants/@-33.8673317,151.1921221,15z/data=!3m1!4b1!4m7!2m6!3m5!2sGoogle+Sydney+-+Pirrama+Road!3s0x6b12ae37b47f5b37:0x8eaddfcd1b32ca52!4m2!1d151.1958561!2d-33.866489?entry=ttu&g_ep=EgoyMDI0MDgyMy4wIKXMDSoASAFQAw%3D%3D")  
                st.write(f"Cuisine: {row['Cuisine']}")
                st.write(f"Price: {row['Price']}")
                st.write(f"Location: {row['Location']}")
                st.write(f"Award: {row['Award']}")
                st.write(f"Phone: {row.get('Phone', 'N/A')}")
                st.map(pd.DataFrame([[row['Latitude'], row['Longitude']]], columns=['lat', 'lon']))

if __name__ == '__main__':
    main()
