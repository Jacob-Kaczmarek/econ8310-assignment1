
import pandas as pd
from prophet import Prophet


train_data_url = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv"
train_data = pd.read_csv(train_data_url, parse_dates=["Timestamp"], index_col="Timestamp")


train_data_prophet = train_data.reset_index().rename(columns={"Timestamp": "ds", "trips": "y"})


model = Prophet()
modelFit = model.fit(train_data_prophet)

test_data_url = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv"
test_data = pd.read_csv(test_data_url, parse_dates=["Timestamp"], index_col="Timestamp")

future = pd.DataFrame({"ds": pd.date_range(start=test_data.index.min(), periods=744, freq="H")})


forecast = model.predict(future)


pred = forecast[["ds", "yhat"]].rename(columns={"yhat": "predicted_trips"})

pred

