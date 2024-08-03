import json
import pickle
import numpy as np

# Global variables
__locations = None 
__data_columns = None
__model = None

def get_estimated_price(location, sqft, bhk, bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    # Fill in other features
    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    return __model.predict([x])[0]

def get_location_names():
    return __locations

def load_saved_artifacts():
    print("Loading saved artifacts...")
    global __data_columns
    global __locations
    global __model

    with open("C:/Users/vamsh/OneDrive/Desktop/MACHINE LEARNING/Real Estate Price Prediction Project/server/artifacts/columns.json", 'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]

    with open("C:/Users/vamsh/OneDrive/Desktop/MACHINE LEARNING/Real Estate Price Prediction Project/server/artifacts/banglore_home_price_model.pickle", 'rb') as f:
        __model = pickle.load(f)
    print("Loading saved artifacts...done")

if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('1st Phase JP Nagar', 1000, 3, 3))
