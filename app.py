from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import plotly.express as px
from MLMODEL import GDPModel  

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/graph', methods=['GET', 'POST'])
def graph():
    if request.method == 'POST':
        # Get form data for prediction
        total_indebtedness = float(request.form['total_indebtedness'])
        investment_rate = float(request.form['investment_rate'])
        jobs_creation = float(request.form['jobs_creation'])
        trade_deficit = float(request.form['trade_deficit'])

        # Make prediction
        prediction = GDPModel().predict([total_indebtedness, investment_rate, jobs_creation, trade_deficit])

        # Add the prediction to the CSV file or your data source
        data = pd.read_csv('dataset_cleaned_tn copy.csv')
        new_row = {'GDP Current prices': prediction,
                   'Total indebtedness': total_indebtedness,
                   'Investment rate': investment_rate,
                   'Jobs creation': jobs_creation,
                   'Trade deficit': trade_deficit}
        #add new row to dataframe
        data = pd.concat([data, pd.DataFrame(new_row, index=[0])], ignore_index=True)
        # data = data.append(new_row, ignore_index=True)
        data.to_csv('dataset_cleaned_tn copy.csv', index=False)

    # Read the data and create the graph
    data = pd.read_csv('dataset_cleaned_tn copy.csv')
    fig = px.scatter(data, x='Year', y='GDP Current prices', title='GDP Prediction Graph')
    graph_html = fig.to_html(full_html=False)

    return render_template('graph.html', graph_html=graph_html)

if __name__ == '__main__':
    # gdp_model = GDPModel()  # Initialize your machine learning model
    app.run(debug=True)
