"""
Predictive Analytics Project
============================

Forecast future sales using historical data.

Machine Learning Model:
Linear Regression

Author:
Rehan Ali
"""


import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# Create output folder

os.makedirs(
    "output",
    exist_ok=True
)



# ----------------------------------
# Load Dataset
# ----------------------------------


try:

    data = pd.read_csv(
        "sales_data.csv"
    )


except FileNotFoundError:


    print(
        "Creating sample sales dataset..."
    )


    np.random.seed(42)


    dates = pd.date_range(

        start="2023-01-01",

        periods=365

    )


    sales = (
        np.random.randint(
            1000,
            5000,
            365
        )
        +
        np.arange(365)*5
    )


    data = pd.DataFrame({

        "Date":dates,

        "Sales":sales

    })


    data.to_csv(

        "sales_data.csv",

        index=False

    )



print("\nDataset Preview")

print(data.head())



# ----------------------------------
# Data Cleaning
# ----------------------------------


data["Date"] = pd.to_datetime(

    data["Date"]

)


data = data.dropna()



# Convert Date into numerical feature

data["Day"] = (

    data["Date"]

    -
    data["Date"].min()

).dt.days




# ----------------------------------
# Visualization Historical Sales
# ----------------------------------


plt.figure(figsize=(10,5))


plt.plot(

    data["Date"],

    data["Sales"]

)


plt.title(
    "Historical Sales Trend"
)


plt.xlabel(
    "Date"
)


plt.ylabel(
    "Sales"
)


plt.xticks(rotation=45)


plt.tight_layout()


plt.savefig(

    "output/historical_sales.png"

)


plt.close()



# ----------------------------------
# Train Test Split
# ----------------------------------


X = data[

    ["Day"]

]


y = data[

    "Sales"

]



X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42

)



# ----------------------------------
# Train Model
# ----------------------------------


model = LinearRegression()


model.fit(

    X_train,

    y_train

)



# ----------------------------------
# Prediction
# ----------------------------------


prediction = model.predict(

    X_test

)



# ----------------------------------
# Model Evaluation
# ----------------------------------


mae = mean_absolute_error(

    y_test,

    prediction

)


rmse = np.sqrt(

    mean_squared_error(

        y_test,

        prediction

    )

)


r2 = r2_score(

    y_test,

    prediction

)



print("\nModel Performance")

print("-----------------")

print(
    "MAE:",
    mae
)


print(
    "RMSE:",
    rmse
)


print(
    "R2 Score:",
    r2
)




# ----------------------------------
# Future Forecast
# ----------------------------------


future_days = pd.DataFrame({

    "Day":

    range(

        data["Day"].max()+1,

        data["Day"].max()+31

    )

})


future_sales = model.predict(

    future_days

)


future_days["Predicted_Sales"] = future_sales



future_days.to_csv(

    "output/sales_prediction.csv",

    index=False

)



# ----------------------------------
# Prediction Visualization
# ----------------------------------


plt.figure(figsize=(10,5))


plt.scatter(

    X_test,

    y_test,

    label="Actual"

)


plt.scatter(

    X_test,

    prediction,

    label="Predicted"

)


plt.title(

    "Actual vs Predicted Sales"

)


plt.xlabel(

    "Days"

)


plt.ylabel(

    "Sales"

)


plt.legend()



plt.savefig(

    "output/prediction_graph.png"

)


plt.close()




# ----------------------------------
# Save Report
# ----------------------------------


with open(

    "output/model_report.txt",

    "w"

) as file:


    file.write(

        "Predictive Analytics Report\n"

    )


    file.write(

        "===========================\n\n"

    )


    file.write(

        f"MAE: {mae}\n"

    )


    file.write(

        f"RMSE: {rmse}\n"

    )


    file.write(

        f"R2 Score: {r2}\n"

    )


    file.write(

        "\nFuture 30 Days Sales Forecast Generated."

    )



print(

    "\nProject Completed Successfully!"

)

print(

    "Check output folder."

)
