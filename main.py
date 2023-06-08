from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib
import pandas as pd
import math
import mysql.connector

app = FastAPI()

class PredictionRequest(BaseModel):
    category: str
    latitude: float
    longitude: float

model_path = "assets/model.h5"
scaler_path = "assets/scaler.save"

# Load the saved model
model = tf.keras.models.load_model(model_path)

# Load the scaler
scaler = joblib.load(scaler_path)

@app.post("/predict")
def predict_endpoint(request: PredictionRequest):
    category = request.category
    latitude = request.latitude
    longitude = request.longitude

    # Make predictions using your existing predict function
    predictions = predict(scaler, model, category, latitude, longitude)

    return {"predictions": predictions}

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth's radius in meters

    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Calculate differences between latitudes and longitudes
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad

    # Apply Haversine formula
    a = math.sin(delta_lat/2) * math.sin(delta_lat/2) + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2) * math.sin(delta_lon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c

    return distance

def getdata(category):
    
    mydb = mysql.connector.connect(
        host="34.101.230.69",
        user="root",
        password="CapstoneBisaYuk@123",
        database="my_database"
    )

    # Create a cursor
    mycursor = mydb.cursor()

    # Make a query that gets all the data from the table 'TOKO' with value of kategori = category
    sql = "SELECT * FROM TOKO WHERE kategori = %s"

    # Execute the query
    mycursor.execute(sql, (category,))

    # Fetch all the data
    myresult = mycursor.fetchall()

    # Put the data into a dataframe
    df = pd.DataFrame(myresult, columns=['id_toko', 'nama_toko', 'kategori', 'latitude', 'longitude', 'deskripsi', 'url_image', 'rerata_rating', 'jumlah_rating'])

    # Select the columns that will be used for the prediction
    df = df[['id_toko', 'latitude', 'longitude', 'rerata_rating', 'jumlah_rating']]

    # Close the connection
    mydb.close()

    return df

def predict(scaler, model, category, latitude, longitude):

    # Get the features
    features = getdata(category)

    # Create a new column for distance
    features['jarak'] = features.apply(lambda row: calculate_distance(latitude, longitude, row['latitude'], row['longitude']), axis=1)

    # Drop the latitude and longitude columns and move jarak to after nama_toko
    features = features.drop(['latitude', 'longitude'], axis=1)
    features = features[['id_toko', 'jarak', 'rerata_rating', 'jumlah_rating']]

    # Drop and save the nama_toko column, jarak, rerata_rating, and jumlah_rating columns to a variable
    dropped_features = features[['id_toko', 'jarak', 'rerata_rating', 'jumlah_rating']]
    features = features.drop(['id_toko'], axis=1)

    # Rename the columns to distance(meters), rating_overall, rating_count
    features = features.rename(columns={'jarak': 'distance(meters)', 'rerata_rating': 'rating_overall', 'jumlah_rating': 'rating_count'})

    # Normalize the features
    features_scaled = scaler.transform(features)

    # Make predictions, return them as a dataframe with the nama_toko column and the predictions column, avoiding NaN values for nama_toko
    predictions = pd.DataFrame(model.predict(features_scaled), columns=['predictions'])
    predictions = pd.concat([dropped_features.reset_index(drop=True), predictions], axis=1)

    # Reformat the predictions column to 2 decimal places
    predictions['predictions'] = predictions['predictions'].apply(lambda x: round(x, 2))

    # Sort the predictions by the predictions column in descending order
    predictions = predictions.sort_values(by=['predictions'], ascending=False)

    # Get the 3 highest predictions and return them as a list of floats
    predictions = predictions['predictions'].head(3).tolist()

    return predictions

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)