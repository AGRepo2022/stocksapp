from flask import Flask, request, render_template_string
import sklearn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import stock_prediction
from keras.models import load_model
from stock_prediction import *
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
    nextsevenday = None
    filename = None
    stocksymbol = None
    str_nextsevenday =None

    if request.method == 'POST':
        if 'stocksymbol' in request.form:
            stocksymbol = request.form['stocksymbol']
            try:
                # Load the model
                model_path = f"/Volumes/iMac2024/2024_nmodes/clustering_2024/news_flask_app/stock_models/{stocksymbol}.h5"
                stock_model_loaded = load_model(model_path)
                
                # Initialize PredictStocks with the stock name
                stockpredictor = PredictStocks(stocksymbol)

                # Use PredictStocks method to predict the next day
                nextday, nextsevenday = stockpredictor.predict_stocks(stock_model_loaded)
                
                 # Plot the graph and save it
                plot_path = stockpredictor.plot_prediction(stocksymbol, nextsevenday)
                
                #string the return
                str_nextsevenday = ', '.join(map(str, nextsevenday))
               
                
                filename = os.path.basename(plot_path)
                print("FILENAME", filename)

                result = True  # Indicate success
            except Exception as e:
                # Handle any errors during prediction
                result = True  # Still render the page
                nextday, nextsevenday = "Error", "Error"
                filename = None

    # Render the template regardless of POST/GET
    return render_template_string(
        template,
        result=result,
        nextday=nextday,
        sevenday=str_nextsevenday,
        filename=filename,
        stocksymbol=stocksymbol
    )


if __name__ == '__main__':
  app.run(host='127.0.0.1', port=5000, debug=True)