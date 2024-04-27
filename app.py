from flask import Flask, render_template, request
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

app = Flask(__name__)

# Load the data from the backend
data = pd.read_csv('sleep.csv')

# Assuming 'Person ID' is not useful for prediction, dropping it
data = data.drop('Person ID', axis=1)

# Handle categorical data (if any)
le = LabelEncoder()
for column in ['Gender', 'BMI Category']:
    data[column] = le.fit_transform(data[column])

# Split the 'Blood Pressure' column into 'Systolic' and 'Diastolic'
data[['Systolic BP', 'Diastolic BP']] = data['Blood Pressure'].str.split('/', expand=True)

# Convert the new columns to numeric values
data[['Systolic BP', 'Diastolic BP']] = data[['Systolic BP', 'Diastolic BP']].apply(pd.to_numeric)

# Drop the original 'Blood Pressure' column
data = data.drop('Blood Pressure', axis=1)

# Define features (X) and target variable (y)
features = ['Age', 'Sleep Duration', 'Quality of Sleep', 'BMI Category', 'Systolic BP', 'Diastolic BP', 'Heart Rate', 'Daily Steps']
X = data[features]
y = data['Sleep Disorder']

# Handle missing values using SimpleImputer for features
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Handle missing values in the target variable
y_imputer = SimpleImputer(strategy='most_frequent')  # You can use a different strategy if needed
y = y_imputer.fit_transform(y.values.reshape(-1, 1)).ravel()

# Create a decision tree classifier
model = DecisionTreeClassifier()

# Train the model
model.fit(X, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form['age'])
        sleep_duration = float(request.form['sleep_duration'])
        quality_of_sleep = int(request.form['quality_of_sleep'])
        bmi_category = le.transform([request.form['bmi_category']])[0]
        systolic_bp = float(request.form['systolic_bp'])
        diastolic_bp = float(request.form['diastolic_bp'])
        heart_rate = int(request.form['heart_rate'])
        daily_steps = int(request.form['daily_steps'])

        new_person_data = {
            'Age': age,
            'Sleep Duration': sleep_duration,
            'Quality of Sleep': quality_of_sleep,
            'BMI Category': bmi_category,
            'Systolic BP': systolic_bp,
            'Diastolic BP': diastolic_bp,
            'Heart Rate': heart_rate,
            'Daily Steps': daily_steps
        }

        new_person_input = pd.DataFrame([new_person_data], columns=features)
        prediction = model.predict(new_person_input)

        # Check if any health metric is outside the desired range for good health
        if sleep_duration < 6:
            well_being_message = "You need to sleep for at least 8 hours for your well-being. Aim to sleep for 8 hours or more."
        elif quality_of_sleep < 7:
            well_being_message = "Ensure you are getting high-quality sleep for your well-being. Aim for a quality of sleep score of 7 or higher."
        elif bmi_category != 1:  # Assuming 1 represents the 'Normal' category
            well_being_message = "Maintaining a normal BMI is important for good health. Consider adopting a balanced diet and regular exercise routine."
        elif systolic_bp < 90 or systolic_bp > 120 or diastolic_bp < 60 or diastolic_bp > 80:
            well_being_message = "Keep your blood pressure within the normal range (90/60 mmHg to 120/80 mmHg) for good cardiovascular health."
        elif heart_rate < 60 or heart_rate > 100:
            well_being_message = "Maintain a healthy heart rate (between 60 and 100 beats per minute) for good cardiovascular fitness."
        else:
            well_being_message = "Your health metrics are within the desired range for good health. Keep up the good work!"

        # Additional recommendations for well-being
        if sleep_duration < 6:
            well_being_message += " Also, consider reducing caffeine intake before bedtime and establishing a relaxing bedtime routine to improve sleep quality."
        elif daily_steps < 8000:
            well_being_message += " Aim to increase your daily steps to at least 8000 by incorporating more walking into your routine. Walking is a great way to stay active and improve cardiovascular health."
        elif daily_steps > 10000:
            well_being_message += " Congratulations on achieving over 10000 steps! Keep up the active lifestyle. Consider trying different forms of exercise to maintain variety and prevent boredom."
        else:
            well_being_message += " Remember to stay hydrated by drinking plenty of water throughout the day. Proper hydration is essential for overall health and well-being. Additionally, consider incorporating meditation or mindfulness practices into your daily routine to reduce stress and improve mental well-being."

        return render_template('result.html', prediction=prediction[0], well_being_message=well_being_message)

if __name__ == '__main__':
    app.run(debug=False)
