from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os
from dotenv import load_dotenv
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from flask_cors import CORS
from model_utils import (
    predict_all,
    predict_growth,
    load_growth_models,
    load_lstm_models
)

load_dotenv()

app = Flask(__name__)
CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://10.130.1.109:3000",
)
CORS_ORIGINS_LIST = [origin.strip() for origin in CORS_ORIGINS.split(",") if origin.strip()]
CORS(app, origins=CORS_ORIGINS_LIST)

# ===== Paths =====
MODEL_DIR = "models"
INFLUX_URL = os.getenv("INFLUX_URL", "http://10.130.1.110:8086")
INFLUX_TOKEN = "U3lsdmFpbk1vbnRhZ255RXN0VW5DaGFtcGlvbl9Gb3JtYXRpb25Mb1JhV0FOX1VuaXZfU2F2b2llXzIwMjMhCg=="
INFLUX_ORG = "training-usmb"
INFLUX_BUCKET = "iot-platform"
MEASUREMENT="stm32"
client = InfluxDBClient(
    url=INFLUX_URL,
    token=INFLUX_TOKEN,
    org=INFLUX_ORG
)

write_api = client.write_api(write_options=SYNCHRONOUS)
query_api = client.query_api()



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
        .field("dissolved_oxygen", float(data.get("dissolved_oxygen")))
        .field("turbidity", float(data.get("turbidity")))
        .field("tds", float(data.get("tds")))
    )

    write_api.write(
        bucket=INFLUX_BUCKET,
        org=INFLUX_ORG,
        record=point
    )

    return jsonify({
        "message": "Data written successfully"
    })

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
      |> sort(columns: ["_time"], desc: true)
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
                "turbidity": record.values.get("turbidity") /10,
                "dissolved_oxygen": record.values.get("dissolved_oxygen"),
                "depth": record.values.get("depth"),
                "tds"   : record.values.get("tds")
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
            print(record.values)
            return jsonify({
                "time": record.get_time().isoformat(),
                "temperature": record.values.get("temperature"),
                "pressure": record.values.get("pressure"),
                "ph": record.values.get("ph"),
                "turbidity": record.values.get("turbidity")/10,
                "dissolved_oxygen": record.values.get("dissolved_oxygen"),
                "depth": record.values.get("depth"),
                "tds"   : record.values.get("tds")
            })

    return jsonify({})

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
                    round(record.values.get("turbidity"), 2) /10
                    if record.values.get("turbidity") is not None
                    else 0,

                "dissolved_oxyen_avg":
                    round(record.values.get("dissolved_oxyen"), 2)
                    if record.values.get("dissolved_oxyen") is not None
                    else 0,

                "depth_avg":
                    round(record.values.get("depth"), 2)
                    if record.values.get("depth") is not None
                    else 0,

                "tds_avg":
                    round(record.values.get("tds"), 2)
                    if record.values.get("tds") is not None
                    else 0
            })

    return jsonify(results)

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
                    round(record.values.get("turbidity"), 2)/10
                    if record.values.get("turbidity") is not None
                    else 0,

                "dissolved_oxygen_avg":
                    round(record.values.get("dissolved_oxygen"), 2)
                    if record.values.get("dissolved_oxygen") is not None
                    else 0,

                "depth_avg":
                    round(record.values.get("depth"), 2)
                    if record.values.get("depth") is not None
                    else 0,

                "tds_avg":
                    round(record.values.get("tds"), 2)
                    if record.values.get("tds") is not None
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
                    round(record.values.get("turbidity"), 2)/10
                    if record.values.get("turbidity") is not None
                    else 0,

                "dissolved_oxygen_avg":
                    round(record.values.get("dissolved_oxygen"), 2)
                    if record.values.get("dissolved_oxygen") is not None
                    else 0,

                "depth_avg":
                    round(record.values.get("depth"), 2)
                    if record.values.get("depth") is not None
                    else 0,
                "tds_avg":
                    round(record.values.get("tds"), 2)
                    if record.values.get("tds") is not None
                    else 0
            })

    return jsonify(results)

@app.route("/predict_length_weight", methods=["POST"])
def predict():
    try:

        data = request.get_json(silent=True) or {}

        return jsonify(
            predict_growth(data)
        )

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500
SEQUENCE_LENGTH = 50
@app.route("/predict_future_values", methods=["POST"])
def predictt():

    try:

        data = request.get_json()

        sequence = np.array(
            data["sequence"],
            dtype=float
        )

        if sequence.shape != (SEQUENCE_LENGTH, 4):

            return jsonify({
                "success": False,
                "message":
                "Expected shape (10,4)"
            }), 400

        prediction = predict_all(sequence)

        return jsonify({
            "success": True,
            "prediction": prediction
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "message": str(e)
        }), 500



@app.route("/read_sequence", methods=["GET"])
def read_sequence():

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
      |> limit(n: {SEQUENCE_LENGTH})
    '''

    tables = query_api.query(query)

    sequence = []

    for table in tables:
        for record in table.records:

            sequence.append([
                float(record.values.get("temperature", 0)),
                float(record.values.get("turbidity", 0)) /10,
                float(record.values.get("dissolved_oxygen", 0)),
                float(record.values.get("ph", 0))
            ])

    if len(sequence) < SEQUENCE_LENGTH:

        return jsonify({
            "success": False,
            "message":
                f"Not enough data points. "
                f"Expected {SEQUENCE_LENGTH}, "
                f"got {len(sequence)}"
        }), 400

    # oldest -> newest
    sequence.reverse()

    return jsonify({
        "success": True,
        "sequence_length": len(sequence),
        "sequence": sequence
    })

if __name__ == "__main__":
    load_lstm_models()
    load_growth_models()
    app.run(debug=True)
