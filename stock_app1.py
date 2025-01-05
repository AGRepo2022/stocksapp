from flask import Flask, request, render_template_string
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import stock_model_training 
import stock_prediction
from tensorflow.keras.models import load_model
from stock_model_training import *
from stock_prediction import *
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

template = '''
<html>
  <body>
    <h1>Stock Prediction</h1>
    <form method="post">
      <label for="stocksymbol">Select Stock Symbol:</label>
      <select id="stocksymbol" name="stocksymbol">
        <option value="AAPL">Apple (AAPL)</option>
        <option value="MSFT">Microsoft (MSFT)</option>
        <option value="TSLA">Tesla (TSLA)</option>
      </select>
      <button type="submit">Submit</button>
    </form>
    {% if result %}
      <h2>Result for selected stock: {{ stocksymbol }}</h2>
      <p>Next day's predicted price: {{ nextday }}</p>
      <p>7-day prediction: {{ sevenday }}</p>
        <p>{{filename}}</p>
            <img src="{{ url_for('static', filename=filename) }}" style="width: 80%; display: block; margin: 0 auto;">
    {% endif %}
  </body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
  result = None
  nextday = None
  sevenday = None
  if request.method == 'POST':
    if 'stocksymbol' in request.form:
      stocksymbol = request.form['stocksymbol']
      print(stocksymbol)
      
    try:
        #load model
        model_path= "/Volumes/iMac2024/2024_nmodes/clustering_2024/news_flask_app/stock_models/" + stocksymbol+".h5"
        stock_model_loaded = load_model(model_path)
        
        
        #Initialize PredictStocks with the stockname
        stockpredictor = PredictStocks(stocksymbol)

        # Use the PredictStocks method to predict the next day
        nextday, nextsevenday = stockpredictor.predict_stocks(stock_model_loaded)
        
        #plot the graph and save it
        plot_path = stockpredictor.plot_prediction(stocksymbol,nextsevenday)
        print("Next day's predicted price:", nextday)
        print(plot_path)

        result = True
        # Return the prediction as a JSON response and include the path to the image
        filename = os.path.basename(plot_path)
        
        return render_template_string(template, result=result, nextday=nextday, sevenday=nextsevenday,filename=filename,\
            stocksymbol=stocksymbol) 
        
    except Exception as e:
        result = True
        nextday, nextsevenday = "Error", "Error"
        filename="None"
        return render_template_string(template, result=result, nextday=nextday, sevenday=nextsevenday,stocksymbol=stocksymbol,filename=filename)
           
      
      
      
    # Here you would put your calculation logic
    # For this example, I'll just use some random numbers
    
   


if __name__ == '__main__':
  app.run(debug=True)