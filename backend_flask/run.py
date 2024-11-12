from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import os
import sys
import pickle
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from app.src.exception import CustomException
from app.pipeline.predict import (
    PredictPipeline,
    JsonToDataframe,
    SignalDataPreprocessing,
    SequenceCreation
)
app = Flask(__name__)

@app.route('/data', methods=['POST'])
def receive_data():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Extract samples from the data
        samples = data.get('samples', [])

        if not samples:
            return jsonify({"error": "No samples received"}), 400

        to_df = JsonToDataframe(data)
        df = to_df.json_to_df()
        preprocess_obj = SignalDataPreprocessing(df)
        preprocessed_df = preprocess_obj.dm()
        seq = SequenceCreation(preprocessed_df)
        seq_df = seq.create_sequences(40)
        pred_pipeline = PredictPipeline()
        prediction = pred_pipeline.predict(seq_df)

        # Return the prediction result as JSON
        return jsonify({"prediction": prediction}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
