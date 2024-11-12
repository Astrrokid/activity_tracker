import os
import sys
import pickle

def load_object(file_path):
    try:
        with open('scaler.pkl', 'rb') as f:
            return pickle.load(f)

    except Exception as e:
        raise CustomException(e, sys)
def json_to_df(data):
    try:
        # Flatten data to have each value on a single level
        flattened_data = []
        for entry in data:
            flattened_entry = {
                "time": entry["time"],
                "ax": entry["accelerometer"]["x"],
                "ay": entry["accelerometer"]["y"],
                "az": entry["accelerometer"]["z"],
                "wx": entry["gyroscope"]["x"],
                "wy": entry["gyroscope"]["y"],
                "wz": entry["gyroscope"]["z"],
            }
            flattened_data.append(flattened_entry)

        # Convert the flattened data into a DataFrame
        return pd.DataFrame(flattened_data)
    except Exception as e:
        raise CustomException(e, sys)
