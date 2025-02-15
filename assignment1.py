
import pandas as pd
from prophet import Prophet


train_data_url = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv"
train_data = pd.read_csv(train_data_url, parse_dates=["Timestamp"], index_col="Timestamp")


train_data_prophet = train_data.reset_index().rename(columns={"Timestamp": "ds", "trips": "y"})


model = Prophet()
modelFit = model.fit(train_data_prophet)

test_data = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv")
test_data['Timestamp'] = pd.to_datetime(test_data['Timestamp'])
future = test_data.rename(columns={'Timestamp': 'ds'})

pred = modelFit.predict(future)


pred = pred[['ds', 'yhat']]
pred['yhat'] = pred['yhat'].astype(int)
pred = pred.set_index('ds')
pred

