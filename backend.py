from flask import Flask, request, render_template
import pandas as pd
from neural_net import model  # Import your prediction function from your model script
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Load the model
model = joblib.load('model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/single')
def single():
    return render_template('single.html')

@app.route('/predict_from_excel', methods=['POST'])
def predict_from_excel():
    # Get the uploaded file
    uploaded_file = request.files['file']

    # Read data from the Excel file
    data = pd.read_excel(uploaded_file)
    def plot_to_base64(plt):
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode()
    # Education_Level = data['Education_Level'].value_counts()
    # labels = Education_Level.index
    # sizes = Education_Level.values
    # plt.figure(figsize=(5,5))
    # plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    # plt.title('Distribution of Education Level')
    # plt.axis('equal')  
    # education_level_pie_chart = plot_to_base64(plt)
    # plt.close()

    # # Pie chart for Income_Category
    # Income_Category = data['Income_Category'].value_counts()
    # labels = Income_Category.index
    # sizes = Income_Category.values
    # plt.figure(figsize=(5, 5))
    # plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    # plt.title('Distribution of Income Category')
    # plt.axis('equal')
    # income_category_pie_chart = plot_to_base64(plt)
    # plt.close()

    # # Bar plot for Income_Category vs Count
    # plt.figure(figsize=(10, 6))
    # data['Income_Category'].value_counts().plot(kind='bar', color='Green')
    # plt.xlabel('Income Category')
    # plt.ylabel('Count')
    # plt.title('Count of Customers by Income Category')
    # plt.xticks(rotation=45, ha='right')
    # plt.tight_layout()
    # income_category_bar_plot = plot_to_base64(plt)
    # plt.close()

    # # Bar plot for Age_Group vs Count
    # # Count the occurrences of each Age_Group
    # age_counts = data['Age_Group'].value_counts()

    # # Sort the age groups based on labels
    # age_counts = age_counts.sort_index()

    # # Plotting
    # plt.figure(figsize=(10, 6))
    # age_counts.plot(kind='bar', color='skyblue')
    # plt.xlabel('Age Group')
    # plt.ylabel('Count')
    # plt.title('Count of Records for Each Age Group')
    # plt.xticks(rotation=45, ha='right')
    # plt.tight_layout()
    # age_group_bar_plot = plot_to_base64(plt)
    # plt.close()
# Pie chart for Education_Level
    Education_Level = data['Education_Level'].value_counts()
    labels = Education_Level.index
    sizes = Education_Level.values
    plt.figure(figsize=(8, 6))  # Adjusted figure size
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title('Distribution of Education Level')
    plt.axis('equal')  
    education_level_pie_chart = plot_to_base64(plt)
    plt.close()

    # Pie chart for Income_Category
    Income_Category = data['Income_Category'].value_counts()
    labels = Income_Category.index
    sizes = Income_Category.values
    plt.figure(figsize=(8, 6))  # Adjusted figure size
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title('Distribution of Income Category')
    plt.axis('equal')
    income_category_pie_chart = plot_to_base64(plt)
    plt.close()

    # Bar plot for Income_Category vs Count
    plt.figure(figsize=(10, 6))  # Adjusted figure size
    data['Income_Category'].value_counts().plot(kind='bar', color='Green')
    plt.xlabel('Income Category')
    plt.ylabel('Count')
    plt.title('Count of Customers by Income Category')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    income_category_bar_plot = plot_to_base64(plt)
    plt.close()

    # Bar plot for Age_Group vs Count
    # Count the occurrences of each Age_Group
    age_counts = data['Age_Group'].value_counts()

    # Sort the age groups based on labels
    age_counts = age_counts.sort_index()

    # Plotting
    plt.figure(figsize=(10, 6))  # Adjusted figure size
    age_counts.plot(kind='bar', color='skyblue')
    plt.xlabel('Age Group')
    plt.ylabel('Count')
    plt.title('Count of Records for Each Age Group')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    age_group_bar_plot = plot_to_base64(plt)
    plt.close()
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])
    print(data)
    
    #Normailizing the features
    scaler = StandardScaler()
    newdata = scaler.fit_transform(data)
    # Pass data to the prediction function
    prediction = model.predict(newdata) 




# Pie chart for Education_Level
    
        # Define a function to determine color based on newdata value
    def get_color(prediction):
        if prediction == 0:
            return 'green'
        else:
            return 'red'

    # Scatter plot for Avg_Utilization_Ratio vs Credit_Limit
    plt.figure(figsize=(10, 6))
    for i in range(len(data)):
        plt.scatter(data['Credit_Limit'][i], data['Avg_Utilization_Ratio'][i], color=get_color(prediction[i]), alpha=0.5)
    plt.title('Avg_Utilization_Ratio vs Credit_Limit')
    plt.xlabel('Credit_Limit')
    plt.ylabel('Avg_Utilization_Ratio')
    plt.grid(True)
    scatter_plot = plot_to_base64(plt)
    plt.close()

    # Scatter plot for Total_Trans_Amt vs Total_Trans_Ct
    plt.figure(figsize=(10, 6))
    for i in range(len(data)):
        plt.scatter(data['Total_Trans_Amt'][i], data['Total_Trans_Ct'][i], color=get_color(prediction[i]), alpha=0.5)
    plt.title('Scatter Plot of Total Transaction Amount vs Total Transaction Count')
    plt.xlabel('Total Transaction Amount')
    plt.ylabel('Total Transaction Count')
    plt.grid(True)
    scatter_plot1 = plot_to_base64(plt)
    plt.close()

    # Scatter plot for Total_Ct_Chng_Q4_Q1 vs Total_Amt_Chng_Q4_Q1
    plt.figure(figsize=(10, 6))
    for i in range(len(data)):
        plt.scatter(data['Total_Ct_Chng_Q4_Q1'][i], data['Total_Amt_Chng_Q4_Q1'][i], color=get_color(prediction[i]), alpha=0.5)
    plt.title('Total_Ct_Chng_Q4_Q1 vs Total_Amt_Chng_Q4_Q1')
    plt.xlabel('Total_Ct_Chng_Q4_Q1')
    plt.ylabel('Total_Amt_Chng_Q4_Q1')
    plt.grid(True)
    scatter_plot2 = plot_to_base64(plt)
    plt.close()

    # Scatter plot for Total_Revolving_Bal vs Total_Trans_Amt
    plt.figure(figsize=(10, 6))
    for i in range(len(data)):
        plt.scatter(data['Total_Revolving_Bal'][i], data['Total_Trans_Amt'][i], color=get_color(prediction[i]), alpha=0.5)
    plt.title('Total_Revolving_Bal vs Total_Trans_Amt')
    plt.xlabel('Total_Revolving_Bal')
    plt.ylabel('Total_Trans_Amt')
    plt.grid(True)
    scatter_plot3 = plot_to_base64(plt)
    plt.close()
    label_map = {0: "LOW RISK", 1: "HIGH RISK"}
    # Convert prediction values using the mapping dictionary
    predicted_labels = [label_map[pred] for pred in prediction]
#  # Pass the entire DataFrame to the prediction function
    # Pass the plots and prediction results to the template
    return render_template('result.html', prediction=predicted_labels,scatter_plot3=scatter_plot3,scatter_plot2=scatter_plot2,education_level_pie_chart=education_level_pie_chart, 
                       income_category_pie_chart=income_category_pie_chart,
                       scatter_plot1=scatter_plot1,income_category_bar_plot=income_category_bar_plot,
                       age_group_bar_plot=age_group_bar_plot,
                       scatter_plot=scatter_plot)

@app.route('/predict_single_values', methods=['POST'])
def predict_single_values():
    # Get form data
    customer_age = int(request.form['customer_age'])
    gender = request.form['gender']
    dependent_count = int(request.form['dependent_count'])
    education_level = request.form['education_level']
    marital_status = request.form['marital_status']
    income_category = request.form['income_category']
    card_category = request.form['card_category']
    months_on_book = int(request.form['months_on_book'])
    total_relationship_count = int(request.form['total_relationship_count'])
    months_inactive_12_mon = int(request.form['months_inactive_12_mon'])
    contacts_count_12_mon = int(request.form['contacts_count_12_mon'])
    credit_limit = float(request.form['credit_limit'])
    total_revolving_bal = float(request.form['total_revolving_bal'])
    avg_open_to_buy = float(request.form['avg_open_to_buy'])
    total_amt_chng_q4_q1 = float(request.form['total_amt_chng_q4_q1'])
    total_trans_amt = float(request.form['total_trans_amt'])
    total_trans_ct = int(request.form['total_trans_ct'])
    total_ct_chng_q4_q1 = float(request.form['total_ct_chng_q4_q1'])
    avg_utilization_ratio = float(request.form['avg_utilization_ratio'])
    age_group = request.form['age_group']
    income_midpoint = request.form['income_midpoint']
    credit_limit_range = request.form['credit_limit_range']
    revolving_balance_range = request.form['revolving_balance_range']

    # Create DataFrame from form data
    data = pd.DataFrame({
        'Customer_Age': [customer_age],
        'Gender': [gender],
        'Dependent_count': [dependent_count],
        'Education_Level': [education_level],
        'Marital_Status': [marital_status],
        'Income_Category': [income_category],
        'Card_Category': [card_category],
        'Months_on_book': [months_on_book],
        'Total_Relationship_Count': [total_relationship_count],
        'Months_Inactive_12_mon': [months_inactive_12_mon],
        'Contacts_Count_12_mon': [contacts_count_12_mon],
        'Credit_Limit': [credit_limit],
        'Total_Revolving_Bal': [total_revolving_bal],
        'Avg_Open_To_Buy': [avg_open_to_buy],
        'Total_Amt_Chng_Q4_Q1': [total_amt_chng_q4_q1],
        'Total_Trans_Amt': [total_trans_amt],
        'Total_Trans_Ct': [total_trans_ct],
        'Total_Ct_Chng_Q4_Q1': [total_ct_chng_q4_q1],
        'Avg_Utilization_Ratio': [avg_utilization_ratio],
        'Age_Group': [age_group],
        'Income_Midpoint': [income_midpoint],
        'Credit_Limit_Range': [credit_limit_range],
        'Revolving_Balance_Range': [revolving_balance_range]
    })
    
    # Make predictions
    # Convert categorical variables to numerical representations using Label Encoding
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])

#Normailizing the features
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
# Make predictions
    prediction = model.predict(data)
    label_map = {0: "LOW RISK", 1: "HIGH RISK"}
    # Convert prediction values using the mapping dictionary
    predicted_labels = [label_map[pred] for pred in prediction]
    return render_template('resultsingle.html', prediction=predicted_labels)

if __name__ == '__main__':
    app.run(debug=True)

