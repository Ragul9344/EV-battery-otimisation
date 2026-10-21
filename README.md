# AI-Based Predictive EV Battery Thermal Management and Optimisation

A beginner-friendly Streamlit student project that trains a Random Forest regression model to predict the next recorded battery-temperature sample and compares it with an explicitly illustrative cooling scenario.

## Features

- Built-in synthetic demo dataset so the app runs immediately.
- Upload compatible CSV data.
- Best-effort support for NASA PCoE Li-ion battery-aging MATLAB .mat files.
- Time-ordered holdout evaluation with mean absolute error (MAE).
- Temperature trend chart, forecast chart, alert threshold and model feature importance.
- Cooling scenario comparison with an adjustable assumed cooling effect.
- Streamlit Community Cloud deployment without installing Python on your own computer.

## Deploy online

1. Open https://share.streamlit.io/ and sign in with GitHub.
2. Choose **Create app** / **Deploy an app**.
3. Select repository **Ragul9344/EV-battery-otimisation**, branch **main**, main file path **app.py**.
4. Deploy. Community Cloud installs packages from requirements.txt.
5. Open the app URL when deployment completes.

Official instructions: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app

## Dataset

See [data/README.md](data/README.md) for NASA download instructions, field descriptions and limitations. The built-in data/demo_battery_thermal.csv file is synthetic; it is not real test data.

## CSV columns

battery_temp_C, current_A, voltage_V, ambient_temp_C, next_temp_C

## Model and method

- Model: scikit-learn RandomForestRegressor.
- Target: the next recorded temperature sample.
- Evaluation: final 20% of rows held out in sequence order; MAE is displayed.
- Cooling scenario: subtracts a user-selected assumed temperature effect. This is not a learned cooling effect or measured cooling-energy optimisation result.

## Limitations and safety

This is an educational decision-support prototype, not a validated battery-management system. It does not control a fan or guarantee safe temperature. NASA's dataset does not directly contain cooling-fan intervention labels. Real cooling optimisation requires pack-specific experiments, cooling-power measurements, defined sampling intervals, validated thermal limits and safety testing. Never bypass an EV's certified BMS or use this dashboard as a safety controller.

## Suggested project title

**Design and Development of an AI-Based Predictive Thermal Management Dashboard for Electric Vehicle Batteries**

## Suggested report sections

1. Abstract and problem statement
2. Literature review and existing system
3. Dataset source and data dictionary
4. System architecture and flowchart
5. Model training and time-ordered evaluation
6. Dashboard and scenario comparison
7. Results, limitations and future hardware integration
8. Conclusion and references

## References

- NASA PCoE Data Set Repository: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
- NASA Li-ion Battery Aging Datasets: https://data.nasa.gov/dataset/li-ion-battery-aging-datasets
- Streamlit Community Cloud deployment: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app
