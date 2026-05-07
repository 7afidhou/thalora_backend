from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])

# ===== Paths =====
MODEL_DIR = "models"
INFLUX_URL = "http://172.16.1.84:8086"
INFLUX_TOKEN = "U3lsdmFpbk1vbnRhZ255RXN0VW5DaGFtcGlvbl9Gb3JtYXRpb25Mb1JhV0FOX1VuaXZfU2F2b2llXzIwMjMhCg=="
INFLUX_ORG = "training-usmb"
INFLUX_BUCKET = "iot-platform"
MEASUREMENT="water_quality"
client = InfluxDBClient(
    url=INFLUX_URL,
    token=INFLUX_TOKEN,
    org=INFLUX_ORG
)

write_api = client.write_api(write_options=SYNCHRONOUS)
query_api = client.query_api()


# ===== Load models =====
# RandomForest
def load_models():
    global rf_len, rf_w, xgb_len, xgb_w, lgb_len, lgb_w, scaler

    with open(os.path.join(MODEL_DIR, "RandomForest_length.pkl"), "rb") as f:
        rf_len = pickle.load(f)

    with open(os.path.join(MODEL_DIR, "RandomForest_weight.pkl"), "rb") as f:
        rf_w = pickle.load(f)

    with open(os.path.join(MODEL_DIR, "XGBoost_length.pkl"), "rb") as f:
        xgb_len = pickle.load(f)

    with open(os.path.join(MODEL_DIR, "XGBoost_weight.pkl"), "rb") as f:
        xgb_w = pickle.load(f)

    with open(os.path.join(MODEL_DIR, "LightGBM_length.pkl"), "rb") as f:
        lgb_len = pickle.load(f)

    with open(os.path.join(MODEL_DIR, "LightGBM_weight.pkl"), "rb") as f:
        lgb_w = pickle.load(f)

    with open(os.path.join(MODEL_DIR, "scaler1.pkl"), "rb") as f:
        scaler = pickle.load(f)

    print("✅ All models + scaler loaded")
# ===== Home route =====
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/add-data", methods=["POST"])
def add_data():

    data = request.json

    point = (
        Point(MEASUREMENT)
        .field("temperature", float(data.get("temperature")))
        .field("pressure", float(data.get("pressure")))
        .field("depth", float(data.get("depth")))
        .field("ph", float(data.get("ph")))
        .field("do", float(data.get("do")))
        .field("turbidity", float(data.get("turbidity")))
    )

    write_api.write(
        bucket=INFLUX_BUCKET,
        org=INFLUX_ORG,
        record=point
    )

    return jsonify({
        "message": "Data written successfully"
    })

# =====================================================
# GET METHOD
# =====================================================


@app.route("/get-data", methods=["GET"])
def get_data():

    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -30d)
      |> filter(fn: (r) => r["_measurement"] == "{MEASUREMENT}")
      |> pivot(
          rowKey: ["_time"],
          columnKey: ["_field"],
          valueColumn: "_value"
      )
    '''

    tables = query_api.query(query)

    results = []

    for table in tables:
        for record in table.records:

            results.append({
                "time": record.get_time().isoformat(),
                "temperature": record.values.get("temperature"),
                "pressure": record.values.get("pressure"),
                "ph": record.values.get("ph"),
                "turbidity": record.values.get("turbidity"),
                "do": record.values.get("do"),
                "depth": record.values.get("depth")
            })

    return jsonify(results)

@app.route("/get-last-sensors-value", methods=["GET"])
def get_last_sensors_value():

    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -30d)
      |> filter(fn: (r) => r["_measurement"] == "{MEASUREMENT}")
      |> pivot(
          rowKey: ["_time"],
          columnKey: ["_field"],
          valueColumn: "_value"
      )
      |> sort(columns: ["_time"], desc: true)
      |> limit(n: 1)
    '''

    tables = query_api.query(query)

    for table in tables:
        for record in table.records:
            return jsonify({
                "time": record.get_time().isoformat(),
                "temperature": record.values.get("temperature"),
                "pressure": record.values.get("pressure"),
                "ph": record.values.get("ph"),
                "turbidity": record.values.get("turbidity"),
                "do": record.values.get("do"),
                "depth": record.values.get("depth")
            })

    return jsonify({})

# =====================================================
# DAILY AVERAGE
# =====================================================

@app.route("/daily-average", methods=["GET"])
def daily_average():

    query = f'''
from(bucket: "{INFLUX_BUCKET}")
  |> range(start: -7d)
  |> filter(fn: (r) => r["_measurement"] == "{MEASUREMENT}")
  |> aggregateWindow(
      every: 1d,
      fn: mean,
      createEmpty: true
  )
  |> group()
  |> pivot(
      rowKey: ["_time"],
      columnKey: ["_field"],
      valueColumn: "_value"
  )
'''

    tables = query_api.query(query)

    results = []

    for table in tables:
        for record in table.records:

            results.append({
                "day": record.get_time().strftime("%A"),
                "date": record.get_time().strftime("%Y-%m-%d"),

                "temperature_avg":
                    round(record.values.get("temperature"), 2)
                    if record.values.get("temperature") is not None
                    else 0,

                "pressure_avg":
                    round(record.values.get("pressure"), 2)
                    if record.values.get("pressure") is not None
                    else 0,

                "ph_avg":
                    round(record.values.get("ph"), 2)
                    if record.values.get("ph") is not None
                    else 0,

                "turbidity_avg":
                    round(record.values.get("turbidity"), 2)
                    if record.values.get("turbidity") is not None
                    else 0,

                "do_avg":
                    round(record.values.get("do"), 2)
                    if record.values.get("do") is not None
                    else 0,

                "depth_avg":
                    round(record.values.get("depth"), 2)
                    if record.values.get("depth") is not None
                    else 0
            })

    return jsonify(results)


# =====================================================
# WEEKLY AVERAGE
# =====================================================

@app.route("/weekly-average", methods=["GET"])
def weekly_average():

    query = f'''
from(bucket: "{INFLUX_BUCKET}")
  |> range(start: -7w)
  |> filter(fn: (r) => r["_measurement"] == "{MEASUREMENT}")
  |> aggregateWindow(
      every: 1w,
      fn: mean,
      createEmpty: true
  )
  |> group()
  |> pivot(
      rowKey: ["_time"],
      columnKey: ["_field"],
      valueColumn: "_value"
  )
'''

    tables = query_api.query(query)

    results = []

    for table in tables:
        for record in table.records:

            results.append({

                "week_start":
                    record.get_time().strftime("%Y-%m-%d"),

                "temperature_avg":
                    round(record.values.get("temperature"), 2)
                    if record.values.get("temperature") is not None
                    else 0,

                "pressure_avg":
                    round(record.values.get("pressure"), 2)
                    if record.values.get("pressure") is not None
                    else 0,

                "ph_avg":
                    round(record.values.get("ph"), 2)
                    if record.values.get("ph") is not None
                    else 0,

                "turbidity_avg":
                    round(record.values.get("turbidity"), 2)
                    if record.values.get("turbidity") is not None
                    else 0,

                "do_avg":
                    round(record.values.get("do"), 2)
                    if record.values.get("do") is not None
                    else 0,

                "depth_avg":
                    round(record.values.get("depth"), 2)
                    if record.values.get("depth") is not None
                    else 0
            })

    return jsonify(results)

@app.route("/monthly-average", methods=["GET"])
def monthly_average():

    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -6mo)
      |> filter(fn: (r) => r["_measurement"] == "{MEASUREMENT}")
      |> aggregateWindow(
          every: 1mo,
          fn: mean,
          createEmpty: true
      )
      |> group()
      |> pivot(
          rowKey: ["_time"],
          columnKey: ["_field"],
          valueColumn: "_value"
      )
    '''

    tables = query_api.query(query)

    results = []

    for table in tables:
        for record in table.records:

            results.append({
                "month": record.get_time().strftime("%B"),
                "year": record.get_time().strftime("%Y"),

                "temperature_avg":
                    round(record.values.get("temperature"), 2)
                    if record.values.get("temperature") is not None
                    else 0,

                "pressure_avg":
                    round(record.values.get("pressure"), 2)
                    if record.values.get("pressure") is not None
                    else 0,

                "ph_avg":
                    round(record.values.get("ph"), 2)
                    if record.values.get("ph") is not None
                    else 0,

                "turbidity_avg":
                    round(record.values.get("turbidity"), 2)
                    if record.values.get("turbidity") is not None
                    else 0,

                "do_avg":
                    round(record.values.get("do"), 2)
                    if record.values.get("do") is not None
                    else 0,

                "depth_avg":
                    round(record.values.get("depth"), 2)
                    if record.values.get("depth") is not None
                    else 0
            })

    return jsonify(results)


# # ===== Prediction route =====
# @app.route("/api/predict", methods=["POST"])
# def predict():
#     try:
#         data = request.get_json(silent=True) or {}
#         print(data)
#         default_inputs = {
#             "Temperature (C)": 0.0,
#             "Turbidity(NTU)": 0.0,
#             "Dissolved Oxygen(g/ml)": 0.0,
#             "PH": 7.0,
#             "Ammonia(g/ml)": 0.0,
#             "Nitrate(g/ml)": 0.0,
#             "Population": 0.0
#         }

#         def get_numeric_value(field_name):
#             raw_value = data.get(field_name, default_inputs[field_name])
#             try:
#                 return float(raw_value)
#             except (TypeError, ValueError):
#                 return default_inputs[field_name]

#         # ===== RAW INPUTS =====
#         temp = get_numeric_value("Temperature (C)")
#         turb = get_numeric_value("Turbidity(NTU)")
#         do   = get_numeric_value("Dissolved Oxygen(g/ml)")
#         ph   = get_numeric_value("PH")
#         nh3  = get_numeric_value("Ammonia(g/ml)")
#         no3  = get_numeric_value("Nitrate(g/ml)")
#         pop  = get_numeric_value("Population")

#         # ===== TRANSFORMS =====
#         pop  = np.log1p(max(pop, 0))
#         nh3  = np.log1p(max(nh3, 0))
#         no3  = np.log1p(max(no3, 0))
#         turb = np.log1p(max(turb, 0))

#         # ===== FEATURE ENGINEERING =====
#         temp_do    = temp * do
#         ammonia_do = nh3 / (do + 1e-6)
#         nitrate_ph = no3 * ph
#         nh3_no3    = nh3 + no3
#         temp_sq    = temp ** 2
#         do_ph      = do * ph

#         features = [
#             temp, turb, do, ph, nh3, no3, pop,
#             temp_do, ammonia_do, nitrate_ph,
#             nh3_no3, temp_sq, do_ph
#         ]

#         features = np.array(features).reshape(1, -1)
#         features_scaled = scaler.transform(features)

#         # ===== PREDICTIONS =====
#         def predict_model(model_len, model_w):
#             return {
#                 "length": float(np.expm1(model_len.predict(features_scaled))[0]),
#                 "weight": float(np.expm1(model_w.predict(features_scaled))[0])
#             }

#         results = {
#             "RandomForest": predict_model(rf_len, rf_w),
#             "XGBoost": predict_model(xgb_len, xgb_w),
#             "LightGBM": predict_model(lgb_len, lgb_w)
#         }

#         return jsonify(results)

#     except Exception as e:
#         return jsonify({"error": str(e)})




if __name__ == "__main__":
    load_models()
    app.run(debug=True)
