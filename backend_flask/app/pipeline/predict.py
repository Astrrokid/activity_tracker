import tensorflow as tf
import os
import sys
import pickle
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from app.src.exception import CustomException
from app.src.utils import load_object
import json
class PredictPipeline:
    def __init__(self):
        pass
    def predict(self, features):
        try:
            self.features = features
            model_path = os.path.join('artifacts', 'optimized_model.tflite')
            preprocessor_path = os.path.join('artifacts', 'scaler.pkl')
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            input_shape = input_details[0]['shape_signature']
            features = features.astype(np.float32).reshape(input_shape[0],input_shape[1],input_shape[2])
            # Set the input tensor
            interpreter.set_tensor(input_details[0]['index'], features)

            # Run the model
            interpreter.invoke()

            # Get the prediction results from the output tensor
            output_data = interpreter.get_tensor(output_details[0]['index'])

            return output_data.flatten().argmax().astype(int), output_data.flatten().max().astype(float)

        except Exception as e:
            raise CustomException(e, sys)



class JsonToDataframe:
    def __init__(self, json_file):
        self.json_file = json_file

    def json_to_df(self):
        try:
            # Flatten data to have each value on a single level
            flattened_data = []
            for entry in self.json_file['samples']:
                flattened_entry = {
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

class SignalDataPreprocessing:
    def __init__(self, df=None):
        try:
            self.df = df
            self.df = self.df.astype(float)
        except Exception as e:
            raise CustomException(e, sys)
    def remove_spike(self, column_name, upper, lower):
        try:
            Q1= self.df[column_name].quantile(lower)
            Q3= self.df[column_name].quantile(upper)
            self.df = self.df[(df[column_name]<=Q3)&(self.df[column_name]>=Q1)]
        except Exception as e:
            raise CustomException(e, sys)

    def lowPassFilter(self, cutoff=2, fs=40, order=4):
        try:
            # Calculate the Nyquist frequency and normal cutoff frequency
            nyquist = 0.5 * fs
            normal_cutoff = cutoff / nyquist
            
            # Create a low-pass filter using the Butterworth filter design
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            
            # Apply the filter to each specified column
            for col in ['ax', 'ay', 'az', 'wx', 'wy', 'wz']:
                self.df[col] = filtfilt(b, a, self.df[col])
        except Exception as e:
            raise CustomException(e, sys)
        
    
    def get_direction_magnitude(self,x):
        try:
            if x['ax']!= 0.0 and x['ay']!=0.0 and x['az']!=0.0 :
                x['acc_magnitude'] = np.sqrt(x['ax']**2 + x['ay']**2 + x['az']**2)
                x['gyro_magnitude'] = np.sqrt(x['wx']**2 + x['wy']**2 + x['wz']**2)
                x['ax_direction']= np.arccos(x['ax']/x['acc_magnitude'])
                x['ay_direction']= np.arccos(x['ay']/x['acc_magnitude'])
                x['az_direction']= np.arccos(x['az']/x['acc_magnitude'])
                
                x['wx_direction']= np.arccos(x['wz']/x['gyro_magnitude'])
                x['wy_direction']= np.arccos(x['wz']/x['gyro_magnitude'])
                x['wz_direction']= np.arccos(x['wz']/x['gyro_magnitude'])
            else:
                x['acc_magnitude']= 0
                x['ax_direction']= 0
                x['ay_direction']= 0
                x['az_direction']= 0
                
                x['gyro_magnitude'] = np.sqrt(x['wx']**2 + x['wy']**2 + x['wz']**2)
                x['wx_direction']= np.arccos(x['wz']/x['gyro_magnitude'])
                x['wy_direction']= np.arccos(x['wz']/x['gyro_magnitude'])
                x['wz_direction']= np.arccos(x['wz']/x['gyro_magnitude'])
            return x
        except Exception as e:
            raise CustomException(e, sys)
        
    def dm(self):
        try:
            self.df = self.df.apply(self.get_direction_magnitude, axis=1) 
            return self.df

        except Exception as e:
            raise CustomException(e, sys)

class SequenceCreation:
    def __init__(self,df):
        self.df = df   
    def create_sequences(self):
        try:

            print (self.df.shape)
            df_np = np.array(self.df)
            num_sequences, num_features = df_np.shape
            sequences= df_np.reshape(-1,num_features)

            preprocessor_path = os.path.join('artifacts', 'scaler.pkl')
            preprocessor = load_object(file_path=preprocessor_path)
            data_scale = preprocessor.transform(sequences)

            sequences = data_scale.reshape(num_sequences, num_features)
            return np.array(sequences)
        
        except Exception as e:
            raise CustomException(e, sys)



class CustomData:
    def __init__(self,
        ax:float,
        ay:float,
        az:float,
        wx:float,
        wy:float,
        wz:float):

        self.ax = ax
        self.ay = ay
        self.az = az
        self.wx = wx
        self.wy = wy
        self.wz = wz

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict={
                'ax': [self.ax],
                'ay': [self.ay],
                'az': [self.az],
                'wx': [self.wx],
                'wy': [self.wy],
                'wz': [self.wz],
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)
def predict_activity(data):
    # Placeholder prediction logic
    return "walking"  # Example response
