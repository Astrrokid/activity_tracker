# Activity Tracking System

This repository contains the code for an activity tracking system that uses accelerometer and gyroscope data to classify physical activities like walking, running, and falling. The system processes raw sensor data, applies preprocessing and filtering techniques, and uses a TensorFlow Lite model for activity prediction.

---

## Features

- **Data Processing:** Converts raw accelerometer and gyroscope data from JSON to a structured DataFrame format.
- **Signal Preprocessing:** Applies low-pass filtering, spike removal, and calculates direction and magnitude of sensor data.
- **Sequence Creation:** Converts preprocessed data into sequences suitable for time-series predictions.
- **Prediction Pipeline:** Uses a TensorFlow Lite model to classify activities and provides confidence scores.
- **API Integration:** Flask API for sending raw sensor data and receiving predictions.

---

## Technologies Used

- **Python:** Main programming language for implementation.
- **TensorFlow Lite:** Lightweight deep learning model for activity prediction.
- **Flask:** API framework for integrating the prediction system.
- **Pandas & NumPy:** Data manipulation and numerical operations.
- **SciPy:** For signal filtering and preprocessing.
- **Pickle:** For saving and loading pre-trained preprocessing objects.

---

## Structure

- `src/` - Contains core components of the system.
  - `exception.py` - Custom exceptions for error handling.
  - `utils.py` - Utility functions for loading models and preprocessing objects.
  - `pipeline/` - Implements the prediction pipeline and preprocessing logic.
- `app.py` - Flask application for API interaction.
- `artifacts/` - Stores pre-trained models and preprocessing artifacts.
- `requirements.txt` - Lists dependencies for the project.

---

## Setup and Installation

### Prerequisites

1. Python 3.x
2. TensorFlow Lite runtime

### Installation Steps

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/activity-tracking-system.git
    cd activity-tracking-system
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Start the Flask application:
    ```bash
    python app.py
    ```

4. The application will run locally at `http://localhost:5000/`.

---

## API Endpoints

### `/data`
- **Method:** `POST`
- **Description:** Accepts raw sensor data and returns the predicted activity.
- **Input:** JSON data containing accelerometer and gyroscope samples.
- **Output:** Predicted activity with a confidence score.
[https://activity-tracker-ml03.onrender.com/data](https://activity-tracker-ml03.onrender.com/data)
### Example Request:
```json
{
    "samples": [
        {
            "time": 123456789,
            "accelerometer": {"x": 0.1, "y": -0.2, "z": 9.8},
            "gyroscope": {"x": 0.01, "y": 0.02, "z": 0.03}
        },
        {
            "time": 123456790,
            "accelerometer": {"x": 0.0, "y": -0.1, "z": 9.7},
            "gyroscope": {"x": 0.02, "y": 0.01, "z": 0.03}
        }
      {
            #38 more datapoints...
      }
    ]
}
```
### Example Response:
```json
{
    "prediction": "Walking",
    "confidence": "0.87"
}
