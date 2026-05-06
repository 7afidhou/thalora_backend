import os
from datetime import datetime, timezone


def _build_client():
    token = os.getenv("INFLUXDB_TOKEN")
    url = os.getenv("INFLUXDB_URL")
    org = os.getenv("INFLUXDB_ORG")
    bucket = os.getenv("INFLUXDB_BUCKET")

    if not all([token, url, org, bucket]):
        return None, None, None

    try:
        from influxdb_client import InfluxDBClient, Point, WritePrecision
    except ModuleNotFoundError:
        return None, None, None

    client = InfluxDBClient(url=url, token=token, org=org)
    return client, (Point, WritePrecision), bucket


def write_prediction(inputs, predictions):
    """
    Write one prediction event to InfluxDB.
    Returns True when written, False when InfluxDB is not configured.
    """
    client_info = _build_client()
    client, influx_classes, bucket = client_info
    if not client:
        return False

    Point, WritePrecision = influx_classes
    org = os.getenv("INFLUXDB_ORG")

    point = (
        Point("fish_prediction")
        .field("temperature_c", float(inputs["Temperature (C)"]))
        .field("turbidity_ntu", float(inputs["Turbidity(NTU)"]))
        .field("dissolved_oxygen_g_ml", float(inputs["Dissolved Oxygen(g/ml)"]))
        .field("ph", float(inputs["PH"]))
        .field("ammonia_g_ml", float(inputs["Ammonia(g/ml)"]))
        .field("nitrate_g_ml", float(inputs["Nitrate(g/ml)"]))
        .field("population", float(inputs["Population"]))
        .field("xgb_length_cm", float(predictions["xgboost"]["length"]))
        .field("xgb_weight_g", float(predictions["xgboost"]["weight"]))
        .field("lstm_length_cm", float(predictions["lstm"]["length"]))
        .field("lstm_weight_g", float(predictions["lstm"]["weight"]))
        .time(datetime.now(timezone.utc), WritePrecision.NS)
    )

    write_api = client.write_api()
    write_api.write(bucket=bucket, org=org, record=point)
    client.close()
    return True
