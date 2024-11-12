from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import os
import sys
import pickle
import numpy as np
import pandas as pd
import json
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
        seq_df = seq.create_sequences()
        pred_pipeline = PredictPipeline()
        predictions = pred_pipeline.predict(seq_df)
        prediction, pred_confidence = predictions
        pred_confidence = json.dumps(pred_confidence.item())
        pred_map = {0: "Walking", 1: "Running", 2: "Falling" }
        activity = pred_map[prediction]
        # Return the prediction result as JSON
        return jsonify({"prediction": activity, "confidence": pred_confidence}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
