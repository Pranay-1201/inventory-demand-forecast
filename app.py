import streamlit as st
import pandas as pd
import joblib

st.title("Retail Inventory Demand Forecasting")

st.write("Upload dataset with mandatory columns: date, store, item, sales")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    required_columns = ['date','store','item','sales']

    if not all(col in df.columns for col in required_columns):
        st.error("Dataset must contain columns: date, store, item, sales")

    else:

        st.success("Dataset uploaded successfully")

        # convert date
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')

        # date features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['weekday'] = df['date'].dt.weekday
        df['weekend'] = (df['weekday'] >= 5).astype(int)

        # required model features
        df['holidays'] = 0
        selected_store = st.selectbox("Select Store", sorted(df['store'].unique()))
        selected_item = st.selectbox("Select Item", sorted(df['item'].unique()))
        filtered_df = df[(df['store'] == selected_store) & (df['item'] == selected_item)]

        last_row = filtered_df.sort_values("date").iloc[-1]

        lag_value = last_row['sales']

        df['lag_7'] = lag_value
        df['lag_30'] = lag_value
        df['rolling_7'] = lag_value
        df['rolling_30'] = lag_value

        # select store and item
        selected_store = st.selectbox("Select Store", sorted(df['store'].unique()), key="store_select")
        selected_item = st.selectbox("Select Item", sorted(df['item'].unique()), key="item_select")

        # forecast horizon
        num = st.number_input("Enter number", min_value=1, max_value=52, value=1)

        unit = st.selectbox(
            "Select time unit",
            ["Days", "Weeks", "Months", "Years"]
        )

        # convert to days
        if unit == "Days":
            forecast_days = num
        elif unit == "Weeks":
            forecast_days = num * 7
        elif unit == "Months":
            forecast_days = num * 30
        else:
            forecast_days = num * 365

        # filter selected store and item
        filtered_df = df[(df['store'] == selected_store) & (df['item'] == selected_item)]

        last_date = filtered_df['date'].max()

        # generate future dates
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=forecast_days
        )

        # create future dataframe
        future_df = pd.DataFrame({
            'date': future_dates,
            'store': selected_store,
            'item': selected_item
        })

        # create date features
        future_df['year'] = future_df['date'].dt.year
        future_df['month'] = future_df['date'].dt.month
        future_df['day'] = future_df['date'].dt.day
        future_df['weekday'] = future_df['date'].dt.weekday
        future_df['weekend'] = (future_df['weekday'] >= 5).astype(int)

        # required model columns
        future_df['holidays'] = 0
        future_df['lag_7'] = lag_value
        future_df['lag_30'] = lag_value
        future_df['rolling_7'] = lag_value
        future_df['rolling_30'] = lag_value

        # model input features
        X_future = future_df[[
            'store','item','day','month','year',
            'weekend','holidays','weekday',
            'lag_7','lag_30','rolling_7','rolling_30'
        ]]

        # load trained model
        model = joblib.load("inventory_demand_model.pkl")

        # predict
        preds = model.predict(X_future)

        # rename predicted column to sales
        future_df['sales'] = preds

        st.write(f"Future demand for Store {selected_store} - Item {selected_item}")

        # show only required columns
        result = future_df[['date','store','item','sales']]
        result['date'] = result['date'].dt.strftime('%d-%m-%Y')

        st.dataframe(result)

        # download predictions
        st.download_button(
            "Download Predictions",
            result.to_csv(index=False),
            "future_predictions.csv"
        )