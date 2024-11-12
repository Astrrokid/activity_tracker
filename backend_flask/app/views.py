# # backend_flask/app/views.py

# from flask import request, jsonify
# from app import app
# from app.models.activity_classifier import predict_activity  # Import the prediction function

# # Define a route for the prediction endpoint
# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         # Extract JSON data from the request
#         data = request.json
        
#         # Call the prediction function and pass in the data
#         prediction = predict_activity(data)
        
#         # Return the prediction as a JSON response
#         return jsonify({"prediction": prediction}), 200
    
#     except Exception as e:
#         # Handle errors and return an error message
#         return jsonify({"error": str(e)}), 400
