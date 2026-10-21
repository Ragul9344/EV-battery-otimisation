# Dataset guide

## NASA Li-ion Battery Aging Dataset (real measurements)

Official NASA PCoE repository page:
https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

Direct ZIP download:
https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip

NASA charge/discharge files can include measured battery voltage, current, temperature, time, and ambient temperature. Original files are MATLAB .mat files, not CSV. NASA does not provide cooling-fan intervention labels in this dataset.

In the Streamlit app, choose **Upload NASA .mat file** and upload a compatible PCoE file such as B0005.mat, B0006.mat, B0007.mat, or B0018.mat. The app attempts to extract charge/discharge measurements and derives the next recorded temperature from sequence order. MATLAB structures can vary; check the app's validation message if a file is unsupported.

NASA documentation:
https://data.nasa.gov/dataset/li-ion-battery-aging-datasets

## Built-in demonstration data

demo_battery_thermal.csv is **synthetic demonstration data**, not recorded from a real EV or NASA testbed. It lets the dashboard run before real data is obtained.

## CSV format for your own measurements

A compatible CSV must include:

| Column | Meaning | Unit |
|---|---|---|
| battery_temp_C | Current recorded battery temperature | °C |
| current_A | Current | A |
| voltage_V | Voltage | V |
| ambient_temp_C | Ambient temperature | °C |
| next_temp_C | Next recorded temperature sample, training target | °C |

Use a consistent sampling interval. Preserve the original source and units in your report. Never label simulated data as experimental measurements.

## Important limitation

The NASA battery aging dataset does not directly label cooling-fan interventions. The cooling effect in this project is a user-adjustable scenario assumption, not a value learned from NASA measurements. This app is for education and research, not real battery control.
