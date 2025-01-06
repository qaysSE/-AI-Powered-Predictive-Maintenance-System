{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "71cb7352-4b8b-427c-9467-34073474f152",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " * Serving Flask app '__main__'\n",
      " * Debug mode: off\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.\n",
      " * Running on all addresses (0.0.0.0)\n",
      " * Running on http://127.0.0.1:5007\n",
      " * Running on http://192.168.1.59:5007\n",
      "Press CTRL+C to quit\n",
      "192.168.1.59 - - [06/Jan/2025 12:17:12] \"GET / HTTP/1.1\" 404 -\n",
      "192.168.1.59 - - [06/Jan/2025 12:17:12] \"GET /api/v1/courses?enrollment_state=active&per_page=1 HTTP/1.1\" 404 -\n",
      "192.168.1.59 - - [06/Jan/2025 12:17:12] \"GET /favicon.ico HTTP/1.1\" 404 -\n",
      "127.0.0.1 - - [06/Jan/2025 12:18:03] \"POST /predict HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [06/Jan/2025 12:18:58] \"POST /predict HTTP/1.1\" 200 -\n"
     ]
    }
   ],
   "source": [
    "from flask import Flask, request, jsonify\n",
    "import joblib\n",
    "import numpy as np\n",
    "\n",
    "# Initialize Flask app\n",
    "app = Flask(__name__)\n",
    "\n",
    "# Load the trained Random Forest model\n",
    "model = joblib.load('predictive_maintenance_rf_model.pkl')\n",
    "\n",
    "# Define the prediction endpoint\n",
    "@app.route('/predict', methods=['POST'])\n",
    "def predict():\n",
    "    # Get JSON data from the request\n",
    "    data = request.json\n",
    "    \n",
    "    # Extract features from the JSON data\n",
    "    features = [\n",
    "        data['engine_temperature'],\n",
    "        data['brake_wear'],\n",
    "        data['tire_pressure'],\n",
    "        data['vibration_level'],\n",
    "        data['battery_health'],\n",
    "        data['mileage']\n",
    "    ]\n",
    "    \n",
    "    # Convert features to a numpy array and reshape for prediction\n",
    "    features_array = np.array(features).reshape(1, -1)\n",
    "    \n",
    "    # Make prediction\n",
    "    prediction = model.predict(features_array)\n",
    "    \n",
    "    # Return the prediction as JSON\n",
    "    return jsonify({'failure_prediction': int(prediction[0])})\n",
    "\n",
    "# Run the Flask app\n",
    "if __name__ == '__main__':\n",
    "    app.run(host='0.0.0.0', port=5007)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.14"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
