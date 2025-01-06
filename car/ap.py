{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "6ed24f1e-ce35-4fad-ab78-3cf443db08c8",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n",
    "from sklearn.model_selection import GridSearchCV\n",
    "import tensorflow as tf\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Dense, Dropout\n",
    "from tensorflow.keras.optimizers import Adam\n",
    "from tensorflow.keras.callbacks import EarlyStopping"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "a4ce0139-5872-4c60-ad4a-88effdfa5a4b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   engine_temperature  brake_wear  tire_pressure  vibration_level  \\\n",
      "0           94.967142    0.321255      32.602703         0.156382   \n",
      "1           88.617357    0.103019      29.859661         1.626021   \n",
      "2           96.476885    0.071808      37.510009         0.367508   \n",
      "3          105.230299    0.094276      26.465687         1.476648   \n",
      "4           87.658466    0.582869      33.660803         1.042893   \n",
      "\n",
      "   battery_health  mileage  failure  \n",
      "0       78.744859    35127        0  \n",
      "1       93.258931    85851        1  \n",
      "2       73.857394   142924        1  \n",
      "3       94.688936    19269        1  \n",
      "4       88.755250    73820        0  \n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "\n",
    "# Set random seed for reproducibility\n",
    "np.random.seed(42)\n",
    "\n",
    "# Number of samples\n",
    "n_samples = 10000\n",
    "\n",
    "# Generate synthetic features\n",
    "data = {\n",
    "    'engine_temperature': np.random.normal(90, 10, n_samples),  # Normal distribution around 90°C\n",
    "    'brake_wear': np.random.uniform(0, 1, n_samples),          # Brake wear level (0 = new, 1 = fully worn)\n",
    "    'tire_pressure': np.random.normal(35, 5, n_samples),       # Tire pressure in PSI\n",
    "    'vibration_level': np.random.exponential(1, n_samples),    # Vibration levels (exponential distribution)\n",
    "    'battery_health': np.random.uniform(70, 100, n_samples),   # Battery health in percentage\n",
    "    'mileage': np.random.randint(0, 200000, n_samples),        # Mileage in kilometers\n",
    "}\n",
    "\n",
    "# Create a DataFrame\n",
    "df = pd.DataFrame(data)\n",
    "\n",
    "# Define failure conditions (target variable)\n",
    "# For example, failure occurs if:\n",
    "# - Engine temperature > 100°C\n",
    "# - Brake wear > 0.8\n",
    "# - Tire pressure < 30 or > 40 PSI\n",
    "# - Vibration level > 2.5\n",
    "# - Battery health < 75%\n",
    "df['failure'] = (\n",
    "    (df['engine_temperature'] > 100) |\n",
    "    (df['brake_wear'] > 0.8) |\n",
    "    (df['tire_pressure'] < 30) | (df['tire_pressure'] > 40) |\n",
    "    (df['vibration_level'] > 2.5) |\n",
    "    (df['battery_health'] < 75)\n",
    ").astype(int)\n",
    "\n",
    "# Save the synthetic dataset to a CSV file (optional)\n",
    "df.to_csv('synthetic_autonomous_vehicle_data.csv', index=False)\n",
    "\n",
    "# Display the first few rows of the dataset\n",
    "print(df.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "6419553b-ad34-4029-9dd3-d9f2c3cc61a4",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "\n",
    "# Load the synthetic dataset\n",
    "df = pd.read_csv('synthetic_autonomous_vehicle_data.csv')\n",
    "\n",
    "# Split into features (X) and target (y)\n",
    "X = df.drop('failure', axis=1)\n",
    "y = df['failure']\n",
    "\n",
    "# Split into training and testing sets\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# Standardize the features\n",
    "scaler = StandardScaler()\n",
    "X_train = scaler.fit_transform(X_train)\n",
    "X_test = scaler.transform(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "7ff502a8-1c04-4117-b2f0-b123567f6e1c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 100.00%\n",
      "Classification Report:\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       1.00      1.00      1.00       667\n",
      "           1       1.00      1.00      1.00      1333\n",
      "\n",
      "    accuracy                           1.00      2000\n",
      "   macro avg       1.00      1.00      1.00      2000\n",
      "weighted avg       1.00      1.00      1.00      2000\n",
      "\n",
      "Confusion Matrix:\n",
      "[[ 667    0]\n",
      " [   0 1333]]\n"
     ]
    }
   ],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n",
    "\n",
    "# Initialize the Random Forest Classifier\n",
    "rf_classifier = RandomForestClassifier(random_state=42)\n",
    "\n",
    "# Train the model\n",
    "rf_classifier.fit(X_train, y_train)\n",
    "\n",
    "# Make predictions\n",
    "y_pred = rf_classifier.predict(X_test)\n",
    "\n",
    "# Evaluate the model\n",
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "print(f\"Accuracy: {accuracy * 100:.2f}%\")\n",
    "print(\"Classification Report:\")\n",
    "print(classification_report(y_test, y_pred))\n",
    "print(\"Confusion Matrix:\")\n",
    "print(confusion_matrix(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "f8c580f5-1336-4a31-9117-1835c032cf8e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/50\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/opt/homebrew/Caskroom/mambaforge/base/lib/python3.10/site-packages/keras/src/layers/core/dense.py:87: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.\n",
      "  super().__init__(activity_regularizer=activity_regularizer, **kwargs)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m200/200\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 1ms/step - accuracy: 0.7135 - loss: 0.5295 - val_accuracy: 0.8544 - val_loss: 0.3311\n",
      "Epoch 2/50\n",
      "\u001b[1m200/200\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 850us/step - accuracy: 0.8559 - loss: 0.3261 - val_accuracy: 0.9013 - val_loss: 0.2335\n",
      "Epoch 3/50\n",
      "\u001b[1m200/200\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 794us/step - accuracy: 0.8805 - loss: 0.2694 - val_accuracy: 0.9200 - val_loss: 0.1861\n",
      "Epoch 4/50\n",
      "\u001b[1m200/200\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 877us/step - accuracy: 0.9066 - loss: 0.2182 - val_accuracy: 0.9312 - val_loss: 0.1559\n",
      "Epoch 5/50\n",
      "\u001b[1m200/200\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 984us/step - accuracy: 0.9228 - loss: 0.1890 - val_accuracy: 0.9544 - val_loss: 0.1242\n",
      "Epoch 6/50\n",
      "\u001b[1m200/200\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 839us/step - accuracy: 0.9342 - loss: 0.1632 - val_accuracy: 0.9556 - val_loss: 0.1065\n",
      "Epoch 7/50\n",
      "\u001b[1m200/200\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 910us/step - accuracy: 0.9372 - loss: 0.1552 - val_accuracy: 0.9650 - val_loss: 0.0978\n",
      "Epoch 8/50\n",
      "\u001b[1m200/200\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 905us/step - accuracy: 0.9421 - loss: 0.1314 - val_accuracy: 0.9719 - val_loss: 0.0820\n",
      "Epoch 9/50\n",
      "\u001b[1m200/200\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 993us/step - accuracy: 0.9448 - loss: 0.1261 - val_accuracy: 0.9700 - val_loss: 0.0798\n",
      "Epoch 10/50\n",
      "\u001b[1m200/200\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 846us/step - accuracy: 0.9519 - loss: 0.1176 - val_accuracy: 0.9856 - val_loss: 0.0599\n",
      "Epoch 11/50\n",
      "\u001b[1m200/200\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 807us/step - accuracy: 0.9587 - loss: 0.0957 - val_accuracy: 0.9731 - val_loss: 0.0654\n",
      "Epoch 12/50\n",
      "\u001b[1m200/200\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 900us/step - accuracy: 0.9635 - loss: 0.0899 - val_accuracy: 0.9781 - val_loss: 0.0572\n",
      "Epoch 13/50\n",
      "\u001b[1m200/200\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 1ms/step - accuracy: 0.9645 - loss: 0.0893 - val_accuracy: 0.9787 - val_loss: 0.0540\n",
      "Epoch 14/50\n",
      "\u001b[1m200/200\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 945us/step - accuracy: 0.9655 - loss: 0.0816 - val_accuracy: 0.9819 - val_loss: 0.0499\n",
      "Epoch 15/50\n",
      "\u001b[1m200/200\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 949us/step - accuracy: 0.9672 - loss: 0.0727 - val_accuracy: 0.9875 - val_loss: 0.0415\n",
      "Epoch 16/50\n",
      "\u001b[1m200/200\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 1ms/step - accuracy: 0.9714 - loss: 0.0666 - val_accuracy: 0.9837 - val_loss: 0.0426\n",
      "Epoch 17/50\n",
      "\u001b[1m200/200\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 870us/step - accuracy: 0.9736 - loss: 0.0621 - val_accuracy: 0.9869 - val_loss: 0.0412\n",
      "Epoch 18/50\n",
      "\u001b[1m200/200\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 877us/step - accuracy: 0.9706 - loss: 0.0634 - val_accuracy: 0.9894 - val_loss: 0.0357\n",
      "Epoch 19/50\n",
      "\u001b[1m200/200\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 893us/step - accuracy: 0.9785 - loss: 0.0554 - val_accuracy: 0.9812 - val_loss: 0.0438\n",
      "Epoch 20/50\n",
      "\u001b[1m200/200\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 1ms/step - accuracy: 0.9789 - loss: 0.0571 - val_accuracy: 0.9856 - val_loss: 0.0370\n",
      "Epoch 21/50\n",
      "\u001b[1m200/200\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 1ms/step - accuracy: 0.9778 - loss: 0.0570 - val_accuracy: 0.9875 - val_loss: 0.0404\n",
      "Epoch 22/50\n",
      "\u001b[1m200/200\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 1ms/step - accuracy: 0.9777 - loss: 0.0528 - val_accuracy: 0.9825 - val_loss: 0.0424\n",
      "Epoch 23/50\n",
      "\u001b[1m200/200\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 882us/step - accuracy: 0.9772 - loss: 0.0548 - val_accuracy: 0.9812 - val_loss: 0.0447\n",
      "Deep Learning Accuracy: 98.45%\n"
     ]
    }
   ],
   "source": [
    "import tensorflow as tf\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Dense, Dropout\n",
    "from tensorflow.keras.optimizers import Adam\n",
    "from tensorflow.keras.callbacks import EarlyStopping\n",
    "\n",
    "# Define the model\n",
    "model = Sequential([\n",
    "    Dense(128, input_dim=X_train.shape[1], activation='relu'),\n",
    "    Dropout(0.2),\n",
    "    Dense(64, activation='relu'),\n",
    "    Dropout(0.2),\n",
    "    Dense(32, activation='relu'),\n",
    "    Dropout(0.2),\n",
    "    Dense(1, activation='sigmoid')\n",
    "])\n",
    "\n",
    "# Compile the model\n",
    "model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])\n",
    "\n",
    "# Train the model\n",
    "history = model.fit(\n",
    "    X_train, y_train,\n",
    "    validation_split=0.2,\n",
    "    epochs=50,\n",
    "    batch_size=32,\n",
    "    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],\n",
    "    verbose=1\n",
    ")\n",
    "\n",
    "# Evaluate the model\n",
    "loss, accuracy = model.evaluate(X_test, y_test, verbose=0)\n",
    "print(f\"Deep Learning Accuracy: {accuracy * 100:.2f}%\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "b19e1f14-a58c-41b9-a61a-9d07560979b3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Random Forest Accuracy: 100.00%\n",
      "Deep Learning Accuracy: 98.45%\n"
     ]
    }
   ],
   "source": [
    "print(f\"Random Forest Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\")\n",
    "print(f\"Deep Learning Accuracy: {accuracy * 100:.2f}%\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "daa76f66-06e5-43bf-80a7-843c9c4023df",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['predictive_maintenance_rf_model.pkl']"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import joblib\n",
    "\n",
    "# Save the Random Forest model\n",
    "joblib.dump(rf_classifier, 'predictive_maintenance_rf_model.pkl')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "7e2e05cd-eae5-45b9-a9bf-9ac3ecc523cd",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Cross-Validation Accuracy: 99.87%\n"
     ]
    }
   ],
   "source": [
    "from sklearn.model_selection import cross_val_score\n",
    "\n",
    "# Perform 5-fold cross-validation\n",
    "cv_scores = cross_val_score(rf_classifier, X_train, y_train, cv=5, scoring='accuracy')\n",
    "print(f\"Cross-Validation Accuracy: {np.mean(cv_scores) * 100:.2f}%\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "fb6847d7-aa49-4d59-b942-56cbc84100d8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA8YAAAIjCAYAAADBbwJEAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjkuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8hTgPZAAAACXBIWXMAAA9hAAAPYQGoP6dpAABU3klEQVR4nO3dB5hU1f0//kMRBKTaIYhg771gVyTYW2zR2I1fe4/RnxUbaOwYS8SIGmPvNXajRsWKHRtYvsGGBREBhfk/n/N9Zv+zy1JdmIX7ej3PuDt37tx75t4Zl/d8zjm3SalUKiUAAAAoqKbVbgAAAABUk2AMAABAoQnGAAAAFJpgDAAAQKEJxgAAABSaYAwAAEChCcYAAAAUmmAMAABAoQnGAAAAFJpgDADTaO+9906LLrpotZsBADQwwRiARmfQoEGpSZMmNbfmzZunLl265GD6v//7v9VuXqM9TpW3448/PjVGZ599drrrrrumad3hw4dP9vWtvfbaM6V9//3vf9Npp52WXnvttdTYlI/Heeedl2ZXDzzwQD6+AI1N82o3AAAm5/TTT0/du3dPY8eOTc8//3wOgs8880x6880309xzz13t5jW641Rp+eWXT401GO+4445pu+22m+bn/P73v09bbLFFrWXzzz//TAvGffv2zT0DVl555ZmyjyKLYPzXv/5VOAYaHcEYgEZr8803T6uvvnr+ff/990/zzTdfOuecc9I999yTdt5552o3r1Eep4b0448/pjZt2qRqW3XVVdMf/vCHNDuLL3datGiRmjYtZme9xvJeApicYv7fGYDZ0vrrr59/fvjhhzXLxo8fn0455ZS02mqrpfbt2+d/fMd6TzzxxGS7of7tb39Liy22WGrZsmVaY4010osvvjjJvqK7b1RdozIdP++8887J/oP/mGOOSV27ds3bW2qppfI+SqVSrfVi34ceemi69dZb07LLLptatWqVevbsmd544438+JVXXpkWX3zxvL+NNtoot7ehPP744/mYxLHp0KFD2nbbbdM777xTa52o4EUb33777bTbbruljh07pvXWW6/m8X/84x/5GEe7O3XqlHbdddf06aef1trG+++/n373u9+lhRZaKL+O3/zmN3m977//vuYYxPG69tpra7pER/f4X+vdd9/NVehoV+w3viSIL08qffPNN+nYY49NK6ywQppnnnlSu3bt8hcKQ4YMqVnnySefzO+HsM8++9S0MXoqhKgi19feOF9xq9xOPO+mm25KJ510Uh4G0Lp16zRq1Kj8+AsvvJA222yz/H6N5RtuuGF69tlnf1V3+uhJcfjhh+dKepzj//mf/8mfje+++y7tueee+XzG7bjjjqv13qz8XFx44YWpW7du+RxHm6JnRkO+l+LYRbU4VHaLL4s2rLPOOmneeefNbYj322233TZJG8qfpfJnND53yy23XHrooYcmWTeGXuy3336pc+fOeb3oWXHQQQflY1MWx+jII4+s+QzH5zC+gJs4ceIMnRNg9qRiDMBsoxwW4x/aZRE2Bg4cmLvb/vGPf0w//PBDuvrqq1OfPn3S4MGDJ+kO+89//jOvE8Eh/oF97rnnph122CF99NFHaa655srrPPzwwzngRYDt169fGjlyZA5KEfQqRcDYZpttcgiPf3zHvv71r3+lP/3pT/kf5BE0Kj399NM5sB1yyCH5fmx7q622ymHlsssuSwcffHD69ttvc5v23XffHEKmRQTPr7/+utayqK6HRx99NAfAHj165MDy008/pQEDBqR11103vfLKK5NMJrbTTjulJZZYInd5Lgeos846K5188sm5Sh+V+6+++ipvY4MNNkivvvpqDkgRNOKYjxs3Lh122GE5HMcxuO+++3LwiBB4/fXX5+evueaa6YADDsjbji8opmbMmDGTvL7YXpyvt956K7+WCJ8xrjoC2y233JK7at9+++1p++23z+vH+Y0gFa8vwtEXX3yRv4yIABgBLoLTMsssk7ulxxct0b7yFzER1mbEGWeckavEEcjjuMTvcU7jfEToO/XUU3MF+ZprrkmbbLJJfn/EsZkR5WMe3cBj2EF8+RPn5T//+U9aZJFF8vmMbsx/+ctfcpiMsFzpuuuuy5+LeG9Gdfviiy/ObYovbhZccMEGeS+tssoquav6I488kt8LdcU+4/O0++675/dTfLEQ24j30JZbbllr3fgi4I477sifmbZt26ZLLrkkf2Y/+eSTHKxD7CuOZ7z/4nwuvfTS+T0ZYTveU3E+4me8B2J5/D8hjlUcsxNOOCGNGDEiXXTRRTN0PoDZUAkAGplrrrkmElnp0UcfLX311VelTz/9tHTbbbeV5p9//lLLli3z/bJffvmlNG7cuFrP//bbb0sLLrhgad99961ZNmzYsLzNeeedt/TNN9/ULL/77rvz8nvvvbdm2corr1xaeOGFS999913Nsocffjiv161bt5pld911V1525pln1tr/jjvuWGrSpEnpgw8+qFkW60Xbox1lV155ZV6+0EILlUaNGlWz/IQTTsjLK9ed0nGq71b5WhZYYIHSyJEja5YNGTKk1LRp09Kee+5Zs+zUU0/Nz/v9739fax/Dhw8vNWvWrHTWWWfVWv7GG2+UmjdvXrP81Vdfzc+/9dZbp9jmNm3alPbaa6/StCifs/puTzzxRF6nV69epRVWWKE0duzYmudNnDixtM4665SWWGKJmmXx+IQJEybZfpyT008/vWbZiy++mLcfx7auOPf1tX3DDTfMt7JoW2yjR48epTFjxtRqV7SpT58++feyWKd79+6l3r17T9Px+Mtf/jLJe6DuNnv27JnfgwceeGCtz8pvfvObWm0tb7NVq1alzz77rGb5Cy+8kJcfddRRDfZeCoccckit92elymMVxo8fX1p++eVLm2yySa3l8fwWLVrU+nxFO2L5gAEDapZFm6JtcU7rKh+rM844I78n33vvvVqPH3/88fl9/8knn9TbVmDOoys1AI3WpptumruGRhfH6Cob1cCouFZWbps1a5YrPyG6PkaX2V9++SV3p40qVl277LJLrYpzuSoYFcUQVaKYkXivvfbKVcmy3r175wpypajAxf6jC2ul6Fod/35/8MEHay3v1atXraraWmutlX9GpSuqXnWXl9s0NdE9NapwlbfK1xJdWKObcdmKK66YX0+0v64DDzyw1v2oysVxjWpxVG3Lt6hORjWw3GW9fKyiYh5VuIYU1b66r2+llVbK5zoqsNG2qHaW2xYV/qheR9fu8izm0UW2PL53woQJeZ3oUh1d3+t7nzSEeA9Fl+CyOBfRpuheHPsvtze6l8d749///vcMd9+NHguV3ZLjPRTvwVheFu/V+FzU976KCntU3cui0hrbKL9HGuK9NDWVxyp6TkRPiPh81nd+4v8Nlb0Noh3RPb782uI4Rg+Brbfeut7x9+VjFUMbYh/x/4TK93dsP94ncU6AYtCVGoBGKwLfkksumf+B/Pe//z3/IzUCTl0xZvX888/PY01//vnnmuV1Z2oO0VWyUjkkxz/Ew8cff5x/Ruirq26IinWjC25lqA3RJbdyW5PbdzlMRvCvb3m5TVMTIaa+f/yX9x/triva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vS0AAACVBOOZ6Mknn0xNmjRJ3333XeEC99TEcbnrrruq3QwAAADBeHY2fvz4ajcBAABgticYT8Uvv/ySDj300NS+ffs033zzpZNPPjmVSqX82PXXX59WX3311LZt27TQQgul3XbbLX355Zf5seHDh6eNN944/96xY8dcId17773z7amnnkoXX3xxXha3WDe8+eabafPNN0/zzDNPWnDBBdMee+yRvv7661qV5mhLVJujLX369En77rtv2mqrrWq1+eeff04LLLBAuvrqq6fpNU6cODEdd9xxqVOnTvl1RJfmSlHx3n///dP888+f2rVrlzbZZJM0ZMiQmsc//PDDtO222+Y2R9vXWGON9Oijj06xW3XYfvvt8+sv3y+L4xrL4pjvuuuu6Ycffpim1wEAADAjBOOpuPbaa1Pz5s3T4MGDc5i94IIL0sCBA2sC6BlnnJFDYnQLjoAbwTd07do13X777fn3oUOHphEjRuTnx61nz57pj3/8Y14Wt1g3wmcEzlVWWSW99NJL6aGHHkpffPFF2nnnnSdpT4sWLdKzzz6brrjiihxYY93YTtl9992XxowZk3bZZZdpfo1t2rRJL7zwQjr33HPT6aefnh555JGax3faaacc+B988MH08ssvp1VXXTX16tUrffPNN/nx0aNHpy222CI99thj6dVXX02bbbZZ2nrrrdMnn3xS7/5efPHF/POaa67J7S7fL4fsOJbxGuIWXyL0799/sm0fN25cGjVqVK0bAADA9Gg+XWsXUITWCy+8MFc2l1pqqfTGG2/k+xFso1pb1qNHj3TJJZfkamkExaicRgU2RPW2Q4cONetGsG3dunWuzpZdeumlORSfffbZNcv+/ve/5/2/9957ackll8zLllhiiRxeK0W7osoaVd9y4IwwG22YFiuuuGI69dRTa7YfbYmQ27t37/TMM8/kLwUiGLds2TKvc9555+Xwetttt6UDDjggrbTSSvlWFl8W3Hnnnemee+7JFe66ovIc4phUHoNy9XrQoEG5Ch+iah5tOeuss+pte79+/VLfvn2n6XUCAADUR8V4KtZee+0cisui2vv++++nCRMm5OppVEYXWWSRHOQ23HDDvM7kKqVTElXnJ554IofZ8m3ppZeuqaKWrbbaapM8N6rGEYZDVJmjslsZ2qclGFdaeOGFa7qER7si6M8777y12jZs2LCadsXjxx57bFpmmWVy2I3H33nnnRk6DtGFuhyK67alPieccEL6/vvva26ffvrpdO8TAAAoNhXjGTR27Ng8xjduN9xwQ66CRhCM+zMyKVaEywjZ55xzziSPRTgsiy7Pde25557p+OOPT88991z6z3/+k7p3757WX3/9ad73XHPNVet+fBEQldtyu2L/McN2XeUqeITi6HodleTFF188tWrVKu24444zdBym1Jb6RBW7XMkGAACYEYLxVMS420rPP/987m787rvvppEjR+bxr9HdOcTY4ErRZTpEdbnu8rrLYtxujEmOimmMaZ4eUc3dbrvtctU4wvE+++yTGkq06/PPP89tqjtJVlmMd46x1TGZVjlMlycUm1IArnsMAAAAqkFX6qmIKvDRRx+dJ9C68cYb04ABA9IRRxyRu09HwI37H330UR5PG2NrK3Xr1i1XPGMSqa+++ioHxhABMwJ3hMeYdToqooccckiezOr3v/99nowquin/61//yiF3WgJkdKeOSbSiC/Nee+3VYK9/0003zd3HI3g//PDDuc1RlT7xxBNrvgiILwruuOOO9Nprr+Wu1zE795SqvOVjEGOHI3R/++23DdZeAACA6SUYT0V0U/7pp5/SmmuumcNrhOKYcCq6TsckUbfeemtadtllc+U4uhJX6tKlS54YKro5x6WMyhNRRdfjZs2a5eeVu2B37tw5V14jBP/2t79NK6ywQr4sU3RXbtq06TQF2OjyHF25Y1sNJYL9Aw88kDbYYIMc0mMSsLiE0scff5xfU4iZuuOSVOuss07uDh5tiErzlJx//vm5+3VU22PSMQAAgGppUipflJfZWlSjI4hHd+oddtghFVVcrimuf9z1yFtS05atq90cgF9teP8tq90EAJits0FM0tuuXbsprmuM8WwuuixHd+yowEZ1eZtttql2kwAAAGYrgvFsLrphxyzUv/nNb3LX7sqJu+Kx6K49OW+//XYeKw0AAFBkgvFsLiaxmlxv+BhrHBNiTU5DjkUGAACYXQnGc7CoHsd1hQEAAJg8s1IDAABQaIIxAAAAhSYYAwAAUGiCMQAAAIUmGAMAAFBogjEAAACFJhgDAABQaIIxAAAAhSYYAwAAUGiCMQAAAIUmGAMAAFBogjEAAACFJhgDAABQaIIxAAAAhSYYAwAAUGiCMQAAAIUmGAMAAFBogjEAAACF1rzaDYCZ4c2+fVK7du2q3QwAAGA2oGIMAABAoQnGAAAAFJpgDAAAQKEJxgAAABSaYAwAAEChCcYAAAAUmmAMAABAoQnGAAAAFJpgDAAAQKEJxgAAABSaYAwAAEChCcYAAAAUmmAMAABAoQnGAAAAFFrzajcAZoblT/1XatqydbWbATPd8P5bVrsJAACzPRVjAAAACk0wBgAAoNAEYwAAAApNMAYAAKDQBGMAAAAKTTAGAACg0ARjAAAACk0wBgAAoNAEYwAAAApNMAYAAKDQBGMAAAAKTTAGAACg0ARjAAAACk0wBgAAoNAEYwAAAApNMAYAAKDQBGMAAAAKTTAGAACg0ARjAAAACk0wBgAAoNAEYwAAAApNMAYAAKDQBGMAAAAKTTCeQaeddlpaeeWVa+7vvffeabvttpulbVh00UXTRRddNMv2t9FGG6UjjzwyzUrVOK4AAECxCMYz6Nhjj02PPfbYLNnXoEGDUocOHSZZ/uKLL6YDDjhglrQBAABgTtW82g2YXc0zzzz59muMHz8+tWjRYoafP//88/+q/QMAAKBiPFl/+9vfUufOndPEiRNrLd92223TvvvuO0lX6rK+ffvmwNquXbt04IEH5vBb2RX50EMPzd2R55tvvtSnT5+8/IILLkgrrLBCatOmTeratWs6+OCD0+jRo/NjTz75ZNpnn33S999/n5o0aZJvse/6ulJ/8sknuX0R2GP/O++8c/riiy9qHi+3+frrr8/Pbd++fdp1113TDz/8MEPHaNy4cbly3qVLl9z2tdZaK7c3jBo1KrVq1So9+OCDtZ5z5513prZt26YxY8bk+59++mluZ1TEO3XqlNs/fPjwGWoPAADAjBCMJ2OnnXZKI0eOTE888UTNsm+++SY99NBDaffdd6/3OdG1+p133snh8MYbb0x33HFHDsqVrr322lwlfvbZZ9MVV1yRlzVt2jRdcskl6a233sqPP/744+m4447Lj62zzjo5/EbQHTFiRL5FGK0rAnyEymjjU089lR555JH00UcfpV122aXWeh9++GG666670n333ZdvsW7//v1n6BhFyH/uuefSTTfdlF5//fV8zDbbbLP0/vvv5/ZutdVW6Z///Get59xwww15zHDr1q3Tzz//nL8ciKD89NNP52MSoT62UfmFwtTCeYTwyhsAAMD0EIwno2PHjmnzzTevFexuu+22XOndeOON631OBN6///3vabnllktbbrllOv3003Pgraw6L7HEEuncc89NSy21VL6FqCDHNqOKu8kmm6Qzzzwz3XLLLTXbjMpuVIoXWmihfKuvC3eE8jfeeCO3d7XVVsvV2+uuuy4H3xiLXBZtiTHLyy+/fFp//fXTHnvsMUNjpaM6fc0116Rbb701b2exxRbLgX299dbLy0N8gRAhvFwdjtB6//3313yxcPPNN+f2DBw4MFfMl1lmmfzc2Ha58jw1/fr1y8enfIuKOwAAwPQQjKcgAtztt9+eq5Llamd0PY4Kb31WWmmlXAkt69mzZ+4SHd2FyyK01vXoo4+mXr165S7JUT2NsBrV6nKgnBZRqY5QWBkMl1122dxFOR4ri/Ad+yhbeOGF05dffpmmV4TwCRMmpCWXXLJmvHXcIohHVTpsscUWaa655kr33HNPvh/HMirJm266ab4/ZMiQ9MEHH+T2lJ8f3anHjh1bs42pOeGEE3I38/Kt8lgDAABMC5NvTcHWW2+dSqVSrnKuscYaubvvhRde+Ku2GWNxK8V42uhyfNBBB6WzzjorB8Nnnnkm7bfffrk7cWXQbggRVCtFJbruOOppEYG/WbNm6eWXX84/K5Ur2lHt3nHHHXMVO75QiJ/Rtbt58+Y124gvCuILhxmdWKxly5b5BgAAMKME4ymYe+650w477JCDW1Q2o+vzqquuOtn1owL6008/5UmnwvPPP59D4pS690awjGB6/vnn11Siy92oyyJgRnV2SqIbclRL41be39tvv52+++67XDluaKusskpuU1Sboyv1lKruvXv3zuOnY+x0dBMvi2MZ3akXWGCBXEkGAACoBl2ppyKCXVSMY+zw5CbdKosKb1R6I5A+8MAD6dRTT80TVE2u63VYfPHF8yRUAwYMyJNlxYzR5Um5Krs/R3U1xgJ//fXX9Xaxju7JMU432vjKK6+kwYMHpz333DNtuOGGafXVV08NLbpQx75iHzHJ2LBhw/I+Y8xvHK+yDTbYII+LjnW7d++exz6XxbIYsx2ThkU1PrYRY4sPP/zw9NlnnzV4mwEAAOojGE9FTIYV3ZuHDh2adttttymuG+OEY3KtCIPRZXibbbapubTS5MS45Lhc0znnnJMnxIrqdITLSjEzdVz6KbYZXYxj8q66okv03XffnScNi/1HUO7Ro0euyM4sMVFWBONjjjkmV9NjtumY6GuRRRap1a7f//73uZpe94uF6Cb+73//O68flfmoescXCzHGWAUZAACYVZqUYhAtzCFi5us8O/WRt6SmLRt2fDY0RsP7b1ntJgAANOpsEJP0Tq3wpmIMAABAoQnGZHHt4MrLLtW9xeMAAABzIrNSk3Xu3Dm99tprU3wcAABgTiQYk8W1hWOGbAAAgKLRlRoAAIBCE4wBAAAoNMEYAACAQpvhYHz99denddddN0/K9PHHH+dlF110Ubr77rsbsn0AAADQ+ILx5Zdfno4++ui0xRZbpO+++y5NmDAhL+/QoUMOxwAAADBHB+MBAwakq666Kp144ompWbNmNctXX3319MYbbzRk+wAAAKDxBeNhw4alVVZZZZLlLVu2TD/++GNDtAsAAAAabzDu3r17eu211yZZ/tBDD6VlllmmIdoFAAAAs0TzGXlSjC8+5JBD0tixY1OpVEqDBw9ON954Y+rXr18aOHBgw7cSAAAAGlMw3n///VOrVq3SSSedlMaMGZN22223PDv1xRdfnHbdddeGbyUAAAA0lmD8yy+/pH/+85+pT58+affdd8/BePTo0WmBBRaYOS0EAACAxjTGuHnz5unAAw/M3ahD69athWIAAACKNfnWmmuumV599dWGbw0AAADMDmOMDz744HTMMcekzz77LK222mqpTZs2tR5fccUVG6p9AAAA0PiCcXmCrcMPP7xmWZMmTfIM1fFzwoQJDddCAAAAaGzBeNiwYQ3fEgAAAJhdgnG3bt0aviUAAAAwuwTj6667boqP77nnnjPaHmgQb/btk9q1a1ftZgAAALOBJqUYGDydOnbsWOv+zz//nK9n3KJFi3z5pm+++aYh2wjTbNSoUal9+/bp+++/F4wBAKDARk1HNpihyzV9++23tW6jR49OQ4cOTeutt1668cYbZ7TdAAAAMMvNUDCuzxJLLJH69++fjjjiiIbaJAAAAMw+wTg0b948/fe//23ITQIAAEDjm3zrnnvuqXU/himPGDEiXXrppWnddddtqLYBAABA4wzG2223Xa37TZo0SfPPP3/aZJNN0vnnn99QbQMAAIDGGYwnTpzY8C0BAACA2WWM8emnn54vz1TXTz/9lB8DAACAOfo6xs2aNctjihdYYIFay0eOHJmXTZgwoSHbCNPMdYwBAIBZch3jyNIxrriuIUOGpE6dOs3IJgEAAKDxjzHu2LFjDsRxW3LJJWuF46gSjx49Oh144IEzo50AAABQ/WB80UUX5Wrxvvvum/r27ZvL0mUtWrRIiy66aOrZs+fMaCcAAABUPxjvtdde+Wf37t3TOuusk+aaa66Z0yoAAABozJdr2nDDDWt+Hzt2bBo/fnytx016BAAAwBwdjONSTccdd1y65ZZb8kzUdZmVmmpb/tR/paYtW1e7GQAAUBjD+2+ZZlczNCv1n/70p/T444+nyy+/PLVs2TINHDgwjznu3Llzuu666xq+lQAAANCYKsb33ntvDsAbbbRR2meffdL666+fFl988dStW7d0ww03pN13373hWwoAAACNpWL8zTffpB49etSMJ477Yb311kv//ve/G7aFAAAA0NiCcYTiYcOG5d+XXnrpPNa4XEnu0KFDw7YQAAAAGlswju7TQ4YMyb8ff/zx6a9//Wuae+6501FHHZXHHwMAAMAcPcY4AnDZpptumt5999308ssv53HGK664YkO2DwAAABpfMK4U1zGOSbfiBgAAAIXoSh3XKT7jjDNSly5d0jzzzJM++uijvPzkk09OV199dUO3EQAAABpXMD7rrLPSoEGD0rnnnptatGhRs3z55ZfP1zQGAACAOToYxzWM//a3v+XrFTdr1qxm+UorrZTHGwMAAMAcHYz/93//N0+0VdfEiRPTzz//3BDtAgAAgMYbjJdddtn09NNPT7L8tttuS6usskpDtAsAAAAa76zUp5xyStprr71y5TiqxHfccUcaOnRo7mJ93333NXwrAQAAoDFUjGP26VKplLbddtt07733pkcffTS1adMmB+V33nknL+vdu/fMaisAAABUt2K8xBJLpBEjRqQFFlggrb/++qlTp07pjTfeSAsuuGDDtwwAAAAaW8U4qsWVHnzwwfTjjz82dJsAAACgcU++NbmgDAAAAHN0MG7SpEm+1V0GAAAAhRhjHBXivffeO7Vs2TLfHzt2bDrwwAPzBFyVYpZqAAAAmOOCcVyiqdIf/vCHhm4PAAAANN5gfM0118y8lgAAAMDsNvkWAAAAzO4EYwAAAApNMJ5JnnzyyTxj93fffVftpgAAADAFgnED2WijjdKRRx5Zc3+dddZJI0aMSO3bt69quwAAAJgywXgmadGiRVpooYUme53nCRMmpIkTJ87ydo0fPz41No2xTQAAQHEIxg0gru381FNPpYsvvjgH4bgNGjSoVlfquN+hQ4d0zz33pGWXXTZfC/qTTz5J48aNS8cee2zq0qVLvh70WmutlbthT4vyNu+66660xBJLpLnnnjv16dMnffrppzXrnHbaaWnllVdOAwcOTN27d8/rhGjX/vvvn+aff/7Url27tMkmm6QhQ4bUPC9+33jjjVPbtm3z46uttlp66aWX8mMff/xx2nrrrVPHjh1zm5dbbrn0wAMP1GpTpWhf5RcEM9omAACAql+uifpFIH7vvffS8ssvn04//fS87K233ppkvTFjxqRzzjknB8J55503LbDAAunQQw9Nb7/9drrppptS586d05133pk222yz9MYbb+SwOzWxzbPOOitdd911uUp98MEHp1133TU9++yzNet88MEH6fbbb0933HFHatasWV620047pVatWqUHH3wwd/e+8sorU69evfLr6NSpU9p9993TKquski6//PL8nNdeey3NNddc+bmHHHJIrvL++9//zsE42j/PPPNM1zGbkTbVJ75YiFvZqFGjpqsdAAAAgnEDiBAXobR169a5+3R49913J1nv559/TpdddllaaaWV8v2oGMe1oeNnhOIQ1eOHHnooLz/77LOnuu/Y5qWXXporzeHaa69NyyyzTBo8eHBac80187IIsRGcoxIbnnnmmfz4l19+mSvX4bzzzsuV3dtuuy0dcMABuU1/+tOf0tJLL50frwzp8djvfve7tMIKK+T7PXr0mO5jNiNtqk+/fv1S3759p3v/AAAAZYLxLBThecUVV6y5H1XhGGu85JJL1lovKqBRUZ4WzZs3T2ussUbN/Qiy0ZX5nXfeqQnG3bp1qwmgIbonjx49epJ9/PTTT+nDDz/Mvx999NG5W/P111+fNt1001zNXWyxxfJjhx9+eDrooIPSww8/nB+LkFz5uqbFjLSpPieccEJua2XFuGvXrtPVFgAAoNgE41kouglXjrWNIBjdiF9++eWa7sRl09s1eUqiu3Ol2O/CCy9c71jm8vjgGAe82267pfvvvz93bT711FNzd+/tt98+B+YYyxyPRTiOqu3555+fDjvssNS0adNUKpUmqWo3RJvqE9XlcoUZAABgRgjGDVgNjurv9IgxvPGc6D68/vrrz9B+f/nllzwpVrk6PHTo0DyJVXSnnpxVV101ff7557navOiii052vahkx+2oo45Kv//973P37gjGIaqyBx54YL5F1faqq67KwTiqwD/88EP68ccfa8JvjE+emmltEwAAQEMzK3UDiTD3wgsvpOHDh6evv/56mi7FFKEzJrnac8898yRUw4YNy+NsowIb1dhpERNiRSCNfUflOWbIXnvttWuCcn2i+3PPnj3Tdtttlyu+0eb//Oc/6cQTT8whO7ovx6RgUb2NGahjIq8XX3yxJmzH9Zr/9a9/5fa+8sor6Yknnqh5LMY6x1jr//f//l/uAv3Pf/4zz1Q9NVNrEwAAwMwiGDeQmDQrukPHpZiiahoTVE2LqMJGMD7mmGPSUkstlYNhhNBFFllkmp4fIfTPf/5z7va87rrr5i7YN9988xSfE9254/JKG2ywQdpnn31yQI+ZrCMEL7jggvl1jBw5MrcrHtt5553T5ptvXjPJVVS5Y2bqCMMxg3asE5OKhZg9+h//+EfefkzOdeONN+Zu2VMztTYBAADMLE1KdQeEMtuISmxUb8vXSub/Jt+KWcK7HnlLatqydbWbAwAAhTG8/5apMWaD77//PrVr126K66oYAwAAUGiCcSMW3Zeja3R9t2m5xjEAAABTZ1bqRmzgwIF5Iqz6xFjeuMVkWwAAAMw4wbgR69KlS7WbAAAAMMfTlRoAAIBCE4wBAAAoNMEYAACAQhOMAQAAKDTBGAAAgEITjAEAACg0wRgAAIBCE4wBAAAoNMEYAACAQhOMAQAAKDTBGAAAgEITjAEAACg0wRgAAIBCE4wBAAAoNMEYAACAQhOMAQAAKDTBGAAAgEITjAEAACi05tVuAMwMb/btk9q1a1ftZgAAALMBFWMAAAAKTTAGAACg0ARjAAAACk0wBgAAoNAEYwAAAApNMAYAAKDQBGMAAAAKTTAGAACg0ARjAAAACk0wBgAAoNAEYwAAAApNMAYAAKDQBGMAAAAKrXm1GwAzw/Kn/is1bdm62s0AYDoM779ltZsAQEGpGAMAAFBogjEAAACFJhgDAABQaIIxAAAAhSYYAwAAUGiCMQAAAIUmGAMAAFBogjEAAACFJhgDAABQaIIxAAAAhSYYAwAAUGiCMQAAAIUmGAMAAFBogjEAAACFJhgDAABQaIIxAAAAhSYYAwAAUGiCMQAAAIUmGAMAAFBogjEAAACFJhgDAABQaIIxAAAAhSYYAwAAUGiCMQAAAIUmGM9iG220UTryyCMbfLunnXZaWnnllRt8uwAAAHM6wRgAAIBCE4wbufHjx1e7CY2C4wAAAMwsgnEV/PLLL+nQQw9N7du3T/PNN186+eSTU6lUyo8tuuii6Ywzzkh77rlnateuXTrggAPy8j//+c9pySWXTK1bt049evTIz/n5558nu48PP/wwrxf7iW2PGzcuHXvssalLly6pTZs2aa211kpPPvnkVNsaz51//vnTbbfdVrMsumwvvPDCNfefeeaZ1LJlyzRmzJh8/7vvvkv7779/fl68hk022SQNGTKkVtu23XbbtOCCC6Z55pknrbHGGunRRx+ttd/JHYe64nWNGjWq1g0AAGB6CMZVcO2116bmzZunwYMHp4svvjhdcMEFaeDAgTWPn3feeWmllVZKr776ag7AoW3btmnQoEHp7bffzs+56qqr0oUXXljv9l9//fW03nrrpd122y1deumlqUmTJjkgP/fcc+mmm27Kj++0005ps802S++///4U2xrP3WCDDWpC9Lfffpveeeed9NNPP6V33303L3vqqadyuI3QHmLbX375ZXrwwQfTyy+/nFZdddXUq1ev9M033+THR48enbbYYov02GOP5dcY7dh6663TJ598Umvf9R2Huvr165e/YCjfunbtOh1nAgAAIKUmpXKpklk2+VaExrfeeiuHznD88cene+65J4feqJSussoq6c4775zidiI0Rsh96aWXaibfuuuuu9Jll12Wttpqq3TiiSemY445Jj8WgTOqx/Gzc+fONdvYdNNN05prrpnOPvvsKe5rwIAB6corr0xvvvlmuvvuu3MYXWihhXKgPfDAA1Pv3r3zds4666xcPd5yyy3za4wqctniiy+ejjvuuMlWfpdffvm8rQjwYVqPQ1SM41YWFeMIx12PvCU1bfl/QR2A2cPw/ltWuwkAzEEiG0Tx7Pvvv8+9UKek+SxrFTXWXnvtmlAcevbsmc4///w0YcKEfH/11Vef5Dk333xzuuSSS3I35Ki4Rnfsuic3gm+E1AiolTNfv/HGG3nb0RW7UgTKeeedd6rt3XDDDdMRRxyRvvrqq1wdjnAfwTiqyPvtt1/6z3/+k0NviC7T0b66240Kc7Q9xOMR5O+///40YsSI/Fri8boV4/qOQ10RvisDOAAAwPQSjBuhGANcKbpA77777qlv376pT58++VuPqBZHmK4UY3qjInzjjTemfffdtyY4RxBt1qxZ7tYcPyvFGN+pWWGFFVKnTp1yKI5bBO8Ixuecc0568cUX81jnddZZp2ZfMf64vvHLHTp0yD9jrPMjjzySq95RSW7VqlXacccdJ5lgq+5xAAAAmBkE4yp44YUXat1//vnn0xJLLDFJaC2Limy3bt1y9+iyjz/+eJL1ImDed999efxuBOiHH344j02OLslRMY7uzeuvv/50tzeq2/G86EYdXcBj/HKMJ46Kc3SxjspuOcTGeOLPP/88j6GO7tD1efbZZ9Pee++dtt9++5owPXz48OluFwAAQEMw+VYVRJfho48+Og0dOjRXd2MMb3RVnpwIzfGcqBJHd+ToUj25sbcRUKOLcgTTzTffPIfO6EIdFeeY4fmOO+5Iw4YNyxN/xVjhWHdaRPfpaGvMSB1V5qZNm+ZJuW644Ybc1bpy3HJ0Dd9uu+1yMI/AG8E+Qn15PHS8nmjHa6+9lrtexyRhEydOnO7jCAAA0BAE4yqIgBpjamPCqkMOOSSH4slNShW22WabdNRRR+WJqSKYRtCc3CzNIYJrzAgd86rFRFg//vhjuuaaa/J+Y0KupZZaKgfX6Aa9yCKLTFObI/xG1TkCcln8XndZVJcfeOCBHJr32WefHMp33XXXXOGOyzOFmIW7Y8eOuft1zEYd1e2oNAMAAFSDWamZI2eeMys1wOzHrNQAVGtWahVjAAAACk0wJo9Fju7X9d2mdo1jAACA2Z1ZqUkDBw7MY57rE5dpAgAAmJMJxqQuXbpUuwkAAABVoys1AAAAhSYYAwAAUGiCMQAAAIUmGAMAAFBogjEAAACFJhgDAABQaIIxAAAAhSYYAwAAUGiCMQAAAIUmGAMAAFBogjEAAACFJhgDAABQaIIxAAAAhSYYAwAAUGiCMQAAAIUmGAMAAFBogjEAAACF1rzaDYCZ4c2+fVK7du2q3QwAAGA2oGIMAABAoQnGAAAAFJpgDAAAQKEJxgAAABSaYAwAAEChCcYAAAAUmmAMAABAoQnGAAAAFJpgDAAAQKEJxgAAABSaYAwAAEChCcYAAAAUmmAMAABAoQnGAAAAFFrzajcAZoblT/1XatqydbWbAdDoDO+/ZbWbAACNjooxAAAAhSYYAwAAUGiCMQAAAIUmGAMAAFBogjEAAACFJhgDAABQaIIxAAAAhSYYAwAAUGiCMQAAAIUmGAMAAFBogjEAAACFJhgDAABQaIIxAAAAhSYYAwAAUGiCMQAAAIUmGAMAAFBogjEAAACFJhgDAABQaIIxAAAAhSYYAwAAUGiCMQAAAIUmGAMAAFBogjEAAACFNscH47333jttt9121W4GAAAAjdQcH4wvvvjiNGjQoFm2v9NOOy2tvPLKqQiefPLJ1KRJk/Tdd99VuykAAAAzrHmaw7Vv377aTZjtjB8/PrVo0WKW7vPnn39Oc8011yzdJwAAQNUrxhMnTkz9+vVL3bt3T61atUorrbRSuu2222pVIx977LG0+uqrp9atW6d11lknDR06tNY2zjzzzLTAAguktm3bpv333z8df/zxtSq2dbtSb7TRRunwww9Pxx13XOrUqVNaaKGFcpW3UlRAY1vzzz9/ateuXdpkk03SkCFDpvp6ojLdt2/fvG60PW7lavXUtlmuNP/9739PiyyySJpnnnnSwQcfnCZMmJDOPffc3M54nWeddVatfcY+Lr/88rT55pvnY9ijR4+aY1j26aefpp133jl16NAhv+Ztt902DR8+fJJjFNvu3LlzWmqppfLy66+/Ph/7OLax/9122y19+eWX+bF4/sYbb5x/79ixY25HbCcsuuii6aKLLqrVhnhtlce53O5tttkmtWnTpuZ13X333WnVVVdNc889d34tcTx/+eWXqR57AACA2TIYRyi+7rrr0hVXXJHeeuutdNRRR6U//OEP6amnnqpZ58QTT0znn39+eumll1Lz5s3TvvvuW/PYDTfckAPVOeeck15++eUcKCNsTc21116bw9gLL7yQQ+fpp5+eHnnkkZrHd9pppxwAH3zwwbzdCGq9evVK33zzzRS3u8suu6RjjjkmLbfccmnEiBH5FsumdZsffvhhfvyhhx5KN954Y7r66qvTlltumT777LN8TOJ1nnTSSbndlU4++eT0u9/9Lgft3XffPe26667pnXfeqanE9unTJ4fbp59+Oj377LM5dG+22Wa5MlwWX0DElw5xHO67776a555xxhl5u3fddVcOw+Xw27Vr13T77bfn3+N58Vqj2/r0iKC8/fbbpzfeeCOf12jfnnvumY444oj09ttvpyuvvDJ/sVD3y4BK48aNS6NGjap1AwAAmC26UkegOfvss9Ojjz6aevbsmZdFhfCZZ57JgeiAAw7IyyIUbbjhhvn3qAZHUBw7dmyuKA4YMCDtt99+aZ999smPn3LKKenhhx9Oo0ePnuK+V1xxxXTqqafm35dYYol06aWX5mDYu3fvvP/BgwfnENuyZcu8znnnnZeDYVRiy+2qT1RsI3RGgI8Ka9m0bjMq6FExjhC77LLL5opshM4HHnggNW3aNFdyIxw/8cQTaa211qrZfoTuqEaHCLIRbuPYXHbZZenmm2/O2x04cGCu0oZrrrkmV4+jKv/b3/42L4svCmKdyi7UlV9CxLm55JJL0hprrJGPb7zOqD6HqGTH9qZXVKDL5668vzjHe+21V80+4/VEdb98vur7ciWqygAAALNdMP7ggw/SmDFjchitFFXMVVZZpVaILVt44YXzzwiYUR2O0BjdjSutueaa6fHHH5/iviu3Wd5uuYtwVEcj+M0777y11vnpp59yRXdGTOs2owtyhOKyBRdcMDVr1iyH4spl5baWlb9YqLz/2muv1ew7jnXldkN8uVC57xVWWGGSccVR2Y6qbmzj22+/zQE7fPLJJzm4/1rRTbtS7Ccq2pUV4uhKHm2N90p0p6/rhBNOSEcffXTN/agYRzUbAACg0QfjclX3/vvvT126dKn1WFRVy6GtckKmcsWzHNBmVN1JnmK75W1GuyIoRzW1rhmpik7PNutr15TaOq37Xm211XK387pivHNZVIwr/fjjj7kLdtziubFuBOK4X9kFuz4R5EulUq1l0S27rrr7jLZG9XeHHXaYZN3oIVCfeK+Uq/AAAACzVTCOimMEmghb5a7SlaalOhtdi1988cU8LrUs7v8aMfb3888/z92ho4I7vaLqGlXOhtzm1Dz//PO1jkHcL1fdY9/RnTq6O8ekX9Pq3XffTSNHjkz9+/evqcDGOO9K5Qpz3dcbITrGHFdWcYcNGzbVfUZboxfA4osvPs3tBAAAmG0n34quvccee2yecCsmw4og/Morr+SxsXF/Whx22GF5gqpY//33388zVL/++us1leUZsemmm+auyDFLc4xXjgmn/vOf/+RJwOoGw/pE8I0QGF2Zv/766zyW+tduc2puvfXWPDb5vffey2NxYzzzoYcemh+Lybjmm2++PBN1TG4VbYvKdczMHZN6TU50VY/gG+fjo48+Svfcc08e71upW7du+VjHZF1fffVVTS+AmHE7ZrSO/cXEWjFmOLqET02MEY/J2KJqHJOxxQRiN910U55wDAAAYI6c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c/DnP/85fwlR/getvwfVPwfB34OZfw6+//77NM8886QWLVqkLbfcMg0YMCD17t07P+ZzUP1zEHwOGofm1W4AUDy77rprze8xOdeKK66YFltssfxtaa9evaraNphV4h89ZfEZiKDcrVu3dMstt6T99tuvqm2b0/Tv3z/ddNNN+f8xMVkOjecc+Hsw87Vt2za99tprafTo0Tl4HX300bmXykYbbVTtphXG1M6Bz0HjoGJMozXffPOlZs2apS+++KLW8ri/0EIL1fucWD6l9cs/p2ebRTYzzkF94o9D7OuDDz5ooJYX+xxUY5tzsll1vGLG0iWXXNLnoIHPwXnnnZdD2cMPP5z/sVnm70H1z0F9/D1o+HMQXX0XX3zxPBvyMccck3bcccfUr1+//JjPQfXPQX18DqpDMKbRiu4mq622Wv5mrWzixIn5fs+ePet9TiyvXD888sgjNet37949/4+rcp1Ro0alF154YbLbLLKZcQ7q89lnn+WxNHGpCH79OajGNudks+p4RSXhww8/9DlowHNw7rnnpjPOOCN3V4/LpVTy96D656A+/h7M/P8XxXPGjRuXf/c5qP45qI/PQZVUe/YvmNqU+DEz4qBBg/I09wcccECeEv/zzz/Pj++xxx6l448/vtalgpo3b14677zz8qUhYtbX+i7XFNu4++67S6+//nqeBdBlCWbdOfjhhx/yZWyee+650rBhw0qPPvpoadVVVy0tscQSpbFjx1btdc5J52DcuHH58ihxW3jhhfPxjt/ff//9ad4mM/8cHHPMMaUnn3wyfw7ic7PpppuW5ptvvtKXX35Zldc4p52D+H99XFLltttuq3UJlPh/UOU6/h5U7xz4ezDzz8HZZ59devjhh0sffvhhXj/+Nsff6KuuuqpmHZ+D6p4Dn4PGQzCm0RswYEBpkUUWyX9cY4r8559/vuaxDTfcME9xX+mWW24pLbnkknn9uEbo/fffX+vxuDTBySefXFpwwQXz/9h69epVGjp06Cx7PUU/B2PGjCn99re/Lc0///w5MMelbOIagAJZw52D+MMa33vWvcV607pNZv452GWXXXJoju116dIl349rXNIw5yD+31LfOYgv68r8PajuOfD3YOafgxNPPLG0+OKLl+aee+5Sx44dSz179szBrpLPQXXPgc9B49Ek/lOtajUAAABUmzHGAAAAFJpgDAAAQKEJxgAAABSaYAwAAEChCcYAAAAUmmAMAABAoQnGAAAAFJpgDAAAQKEJxgAAABSaYAwAc4i99947NWnSZJLbBx980CDbHzRoUOrQoUOqpniN2223XWqshg8fno/5a6+9Vu2mADAdmk/PygBA47bZZpula665ptay+eefPzU2P//8c5prrrnSnGT8+PHVbgIAM0jFGADmIC1btkwLLbRQrVuzZs3yY3fffXdaddVV09xzz5169OiR+vbtm3755Zea515wwQVphRVWSG3atEldu3ZNBx98cBo9enR+7Mknn0z77LNP+v7772sq0aeddlp+LH6/6667arUjKstRYa6sot58881pww03zPu/4YYb8mMDBw5MyyyzTF629NJLp8suu2y6Xu9GG22UDjvssHTkkUemjh07pgUXXDBdddVV6ccff8ztbdu2bVp88cXTgw8+WPOceC3Rnvvvvz+tuOKKed9rr712evPNN2tt+/bbb0/LLbdcPqaLLrpoOv/882s9HsvOOOOMtOeee6Z27dqlAw44IHXv3j0/tsoqq+R9RPvCiy++mHr37p3mm2++1L59+3wcXnnllVrbi/XjeGy//fapdevWaYkllkj33HNPrXXeeuuttNVWW+X9xWtbf/3104cffljz+K89ngBFJRgDQAE8/fTTOcAdccQR6e23305XXnllDq5nnXVWzTpNmzZNl1xySQ5f1157bXr88cfTcccdlx9bZ5110kUXXZQD2YgRI/Lt2GOPna42HH/88Xn/77zzTurTp08Ox6ecckpuQyw7++yz08knn5z3PT1i/QicgwcPziH5oIMOSjvttFNuc4TP3/72t2mPPfZIY8aMqfW8P/3pTznsRmiNqvrWW2+dK9nh5ZdfTjvvvHPadddd0xtvvJG/BIi2lcN+2XnnnZdWWmml9Oqrr+bHow3h0UcfzcfojjvuyPd/+OGHtNdee6VnnnkmPf/88zn0brHFFnl5pfiyIvb7+uuv58d333339M033+TH/vd//zdtsMEGOajHuYk27rvvvjVfbjTU8QQopBIAMEfYa6+9Ss2aNSu1adOm5rbjjjvmx3r16lU6++yza61//fXXlxZeeOHJbu/WW28tzTvvvDX3r7nmmlL79u0nWS/+OXHnnXfWWhbrxfph2LBheZ2LLrqo1jqLLbZY6Z///GetZWeccUapZ8+eU3yN2267bc39DTfcsLTeeuvV3P/ll1/y695jjz1qlo0YMSLv/7nnnsv3n3jiiXz/pptuqlln5MiRpVatWpVuvvnmfH+33XYr9e7du9a+//SnP5WWXXbZmvvdunUrbbfddrXWKb/WV199tTQlEyZMKLVt27Z077331iyL55100kk190ePHp2XPfjgg/n+CSecUOrevXtp/Pjx9W5zRo4nAP/HGGMAmINsvPHG6fLLL6+5H92iw5AhQ9Kzzz5bq0I8YcKENHbs2FxJja67UeXs169fevfdd9OoUaNyJbLy8V9r9dVXr/k9ujpHF+D99tsv/fGPf6xZHvuMrsbTI7pDl0W38XnnnTd3CS+L7tXhyy+/rPW8nj171vzeqVOntNRSS+VKa4if2267ba3111133Vw1j+NW7p5e+Zqm5IsvvkgnnXRS7sYd7YhtxHH95JNPJvta4txFhb7c7pjQK7pO1zc2uyGPJ0ARCcYAMAeJMBVjauuKscLRTXeHHXaY5LEYjxrjgGPsanRDjvAcQTG6/UbQikmlphSMY2zs/xU8/3/lLsl121bZnhDjgddaa61a65VD57SqGxSjPZXL4n6YOHFiamiVr2lKohv1yJEj08UXX5y6deuWu0NHMK87YVd9r6Xc7latWk12+w15PAGKSDAGgAKISbeGDh1ab2gOMV41AliMuY2xxuGWW26ptU6LFi1ypbOuGJ8b42nL3n///UnG89YVVdzOnTunjz76KI+jrYYY67vIIovk37/99tv03nvv5YmrQvyMCnuluL/kkktOMWjGMQp1j1M8NybCinHD4dNPP01ff/31dLU3qskxXri+Gb0bw/EEmJ0JxgBQADEpU1SEIwjuuOOOOfxG9+qYifnMM8/MgTkC14ABA/IkVBHkrrjiiklmYY7K5GOPPZYnnIoqctw22WSTdOmll+YKaATCP//5z9N0KaaoYB9++OG5q29cZmrcuHHppZdeyiH16KOPTjPb6aefnrtdR6g88cQT8wRe5WskH3PMMWmNNdbIs07vsssu6bnnnsuvcWqzPC+wwAK5svvQQw+l3/zmN7kaH68vJtu6/vrrc9fr6KYeE39NqQJcn0MPPTSfn5gQ7IQTTsjbjXC/5ppr5m7g1T6eALMzs1IDQAHELND33Xdfevjhh3Pgi8sTXXjhhblbb4igG5drOuecc9Lyyy+fZziO8caVYpbnAw88MAfFqBKfe+65eXlUmePyTjH+dbfddsuzVU/LmOT9998/X14orrscY4LjEkYx63P5kkczW//+/fMs2auttlr6/PPP07333ltT8Y0Ke1TMb7rppnw84ouFCNJ77733FLfZvHnzPLN3zPodFdzyOOWrr746B9TYbsyQHQE2QvT0iBAfs1HHlxNxrKLd0XW6/CVEtY8nwOysSczAVe1GAADMKjEBVkxSFkE1rrcMACrGAAAAFJpgDAAAQKHpSg0AAEChqRgDAABQaIIxAAAAhSYYAwAAUGiCMQAAAIUmGAMAAFBogjEAAACFJhgDAABQaIIxAAAAqcj+P68D5wzEThmkAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 1000x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Get feature importances\n",
    "importances = rf_classifier.feature_importances_\n",
    "feature_names = X.columns\n",
    "\n",
    "# Plot feature importances\n",
    "plt.figure(figsize=(10, 6))\n",
    "plt.barh(feature_names, importances)\n",
    "plt.xlabel('Feature Importance')\n",
    "plt.ylabel('Feature')\n",
    "plt.title('Random Forest Feature Importance')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "974d0cce-37a1-48c1-8509-5fc1ccf7fc1a",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-01-06 12:09:07.474 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-06 12:09:07.779 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run /opt/homebrew/Caskroom/mambaforge/base/lib/python3.10/site-packages/ipykernel_launcher.py [ARGUMENTS]\n",
      "2025-01-06 12:09:07.779 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-06 12:09:07.779 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-06 12:09:07.780 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-06 12:09:07.780 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-06 12:09:07.781 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-06 12:09:07.781 Session state does not function when running a script without `streamlit run`\n",
      "2025-01-06 12:09:07.781 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-06 12:09:07.782 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-06 12:09:07.782 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-06 12:09:07.783 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-06 12:09:07.783 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-06 12:09:07.783 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-06 12:09:07.784 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-06 12:09:07.784 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-06 12:09:07.784 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-06 12:09:07.785 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-06 12:09:07.785 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-06 12:09:07.785 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-06 12:09:07.786 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-06 12:09:07.786 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-06 12:09:07.786 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-06 12:09:07.786 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-06 12:09:07.787 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-06 12:09:07.787 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-06 12:09:07.787 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-06 12:09:07.787 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-06 12:09:07.788 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-06 12:09:07.788 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-06 12:09:07.788 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-06 12:09:07.789 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-06 12:09:07.789 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-06 12:09:07.790 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-06 12:09:07.790 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-06 12:09:07.791 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-06 12:09:07.792 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-06 12:09:07.793 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-06 12:09:07.793 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-06 12:09:07.794 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-06 12:09:07.795 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-06 12:09:07.796 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-06 12:09:07.796 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-06 12:09:07.797 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-06 12:09:07.797 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": []
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
