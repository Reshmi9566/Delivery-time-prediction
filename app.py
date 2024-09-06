from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model and scaler
with open("model.pkl", 'rb') as model_file:
    model = pickle.load(model_file)


@app.route('/')
def home():
    return render_template("in.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    delivery_person_age = float(request.form['age'])
    delivery_person_ratings = float(request.form['ratings'])
    weather_conditions = request.form['weather']
    road_traffic_density = request.form['traffic']
    vehicle_condition = int(request.form['vehicle'])
    type_of_order = request.form['order_type']
    type_of_vehicle = request.form['vehicle_type']
    multiple_deliveries = int(request.form['multiple_deliveries'])
    festival = request.form['festival']
    city = request.form['city']
    order_preparation_time = float(request.form['prep_time'])
    
    distance = float(request.form['distance'])
    
    # New fields for order date and time
    order_date = request.form['order_date']
    order_time = request.form['order_time']
    order_datetime = pd.to_datetime(order_date + ' ' + order_time, format='%Y-%m-%d %H:%M')
    
    # Extract order hour and order minute
    order_hour = order_datetime.hour
    order_minute = order_datetime.minute
    order_day = order_datetime.day
    def categorize_time_of_day(time):
        if time.hour >= 5 and time.hour < 12:
           return 'Morning'
        elif time.hour >= 12 and time.hour < 17:
           return 'Afternoon'
        elif time.hour >= 17 and time.hour < 21:
           return 'Evening'
        else:
           return 'Night'

    time_of_day = categorize_time_of_day(order_datetime)
    # Create DataFrame for new data
    new_data = pd.DataFrame({
        'Delivery_person_Age': [delivery_person_age],
        'Delivery_person_Ratings': [delivery_person_ratings],
        'Weatherconditions': [weather_conditions],
        'Road_traffic_density': [road_traffic_density],
        'Vehicle_condition': [vehicle_condition],
        'Type_of_order': [type_of_order],
        'Type_of_vehicle': [type_of_vehicle],
        'multiple_deliveries': [multiple_deliveries],
        'Festival': [festival],
        'City': [city],
        'Order_Preparation_Time': [order_preparation_time],
        
        'distance_km': [distance],
        'order_day' : [order_day],
        'order_hour' : [order_hour],
        'order_minute' : [order_minute],
        'Time_of_Day' : [time_of_day]
        
    })

    # Concatenate 'order_date' and 'order_time' element-wise
    #new_data['order_datetime'] = new_data['order_date'].astype(str) + ' ' + new_data['order_time'].astype(str)

# Convert 'order_datetime' column to datetime
    #new_data['order_datetime'] = pd.to_datetime(new_data['order_datetime'], format='%Y-%m-%d %H:%M')

# Function to categorize time of day
    

# Apply the function to the 'order_datetime' column
    #new_data['Time_of_Day'] = new_data['order_datetime'].apply(categorize_time_of_day)

    #new_data.drop('order_datetime',axis=1,inplace=True)

    # One-hot encode categorical features
    new_data = pd.get_dummies(new_data, columns=[
        'Weatherconditions', 'Road_traffic_density', 'Vehicle_condition',
        'Type_of_order', 'Type_of_vehicle', 'multiple_deliveries', 'Festival',
        'City', 'Time_of_Day', 'order_day'
    ])

    # Ensure the new data has the same columns as the training data
    columns = columns = ['Delivery_person_Age', 'Delivery_person_Ratings', 'Vehicle_condition',
       'multiple_deliveries', 'Order_Preparation_Time',
       'distance_km', 'order_day', 'order_hour', 'order_minute',
       'Weatherconditions_Cloudy', 'Weatherconditions_Fog',
       'Weatherconditions_Sandstorm', 'Weatherconditions_Stormy',
       'Weatherconditions_Sunny', 'Weatherconditions_Windy',
       'Road_traffic_density_High', 'Road_traffic_density_Jam',
       'Road_traffic_density_Low', 'Road_traffic_density_Medium',
       'Type_of_order_Buffet', 'Type_of_order_Drinks', 'Type_of_order_Meal',
       'Type_of_order_Snack', 'Type_of_vehicle_electric_scooter',
       'Type_of_vehicle_motorcycle', 'Type_of_vehicle_scooter', 'Festival_No',
       'Festival_Yes', 'City_Metropolitian', 'City_Semi-Urban', 'City_Urban',
       'Time_of_Day_Afternoon', 'Time_of_Day_Evening', 'Time_of_Day_Morning',
       'Time_of_Day_Night']
    missing_cols = set(columns) - set(new_data.columns)
    for col in missing_cols:
        new_data[col] = 0

    # Ensure the order of columns matches the training data
    new_data = new_data[columns]

    

    # Predict the outcome
    prediction = model.predict(new_data)
    prediction_result = int(prediction[0])

    return render_template("home.html", prediction_result=prediction_result)

if __name__ == "__main__":
    app.run(debug=True)
