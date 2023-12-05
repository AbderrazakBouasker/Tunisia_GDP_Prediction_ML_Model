import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.model_selection import cross_val_score,LeaveOneOut
from datetime import datetime
import pickle

class GDPModel:
      def __init__(self):
            data = pd.read_csv("dataset_tn.csv")
            df = pd.DataFrame(data)
            df=data.iloc[:,1:]
            df=data.iloc[:,:]
            df['Year'] = pd.to_datetime(df['Year'], format='%d/%m/%Y')
            df.index = df['Year']
            del df['Year']
            df2=data
            df2.dropna(axis=0, inplace=True)
            df3=df2.iloc[:,1:]
            df3['Year'] = pd.to_datetime(df2['Year'], format='%d/%m/%Y')
            df3.index = df2['Year']
            del df3['Year']
            # Min-Max Scaling for GDP Current Prices
            self.scaler_gdp = MinMaxScaler()
            df3['GDP Current prices'] = self.scaler_gdp.fit_transform(df3[['GDP Current prices']])

            # Robust Scaling for Total Indebtedness
            self.scaler_indebtedness = RobustScaler()
            df3['Total indebtedness'] = self.scaler_indebtedness.fit_transform(df3[['Total indebtedness']])

            # Standardization for Investment Rate
            self.scaler_investment = StandardScaler()
            df3['Investment rate'] = self.scaler_investment.fit_transform(df3[['Investment rate']])

            # Min-Max Scaling for Jobs Creation
            self.scaler_jobs = MinMaxScaler()
            df3['Jobs creation'] = self.scaler_jobs.fit_transform(df3[['Jobs creation']])

            # Robust Scaling for Trade Deficit
            self.scaler_trade_deficit = RobustScaler()
            df3['Trade deficit'] = self.scaler_trade_deficit.fit_transform(df3[['Trade deficit']])
            y=df3['GDP Current prices']
            X=df3.loc[:,'Total indebtedness':'Trade deficit']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
            self.model = LinearRegression()
            self.model.fit(X_train, y_train)
            #save model as pickle file 
            with open('model.pkl', 'wb') as f:
                pickle.dump(self.model, f)
            
            
      def predict(self, features):
            #open model from pickle file
            with open('model.pkl', 'rb') as f:
                  self.model = pickle.load(f)
            data1 = [features]
            columns1 = ['Total indebtedness', 'Investment rate', 'Jobs creation', 'Trade deficit']
            x_test = pd.DataFrame(data1, columns=columns1)
            # Robust Scaling for Total Indebtedness
            x_test['Total indebtedness'] = self.scaler_indebtedness.transform(x_test[['Total indebtedness']])

            # Standardization for Investment Rate
            x_test['Investment rate'] = self.scaler_investment.transform(x_test[['Investment rate']])

            # Min-Max Scaling for Jobs Creation
            x_test['Jobs creation'] = self.scaler_jobs.transform(x_test[['Jobs creation']])

            # Robust Scaling for Trade Deficit
            x_test['Trade deficit'] = self.scaler_trade_deficit.transform(x_test[['Trade deficit']])
            prediction = self.model.predict(x_test)
            #reverse normalization
            prediction = self.scaler_gdp.inverse_transform(prediction.reshape(-1, 1))
            return prediction.flatten()[0]
            
                