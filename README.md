# ActivityPredictionApp

A cross-platform mobile app for predicting activities (walking, running, falling) using Flutter for the mobile frontend and Flask for the backend API server.

## Project Structure
- **backend_flask**: Flask API server for handling predictions.
- **frontend_flutter**: Flutter app for mobile device interface.
- **assets/models/model.tflite**: TensorFlow Lite model used for predictions.

## Instructions
1. **Backend Setup**:
   - Navigate to `backend_flask` and install dependencies:
     ```
     cd backend_flask
     pip install -r requirements.txt
     ```
   - Run the Flask server:
     ```
     python run.py
     ```

2. **Frontend Setup**:
   - Navigate to `frontend_flutter` and run the Flutter app:
     ```
     cd frontend_flutter
     flutter run
     ```
