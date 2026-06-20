import json
import random
import time
import paho.mqtt.client as mqtt

client = mqtt.Client()
client.connect("localhost", 1883)

while True:
    payload = {
        "deviceInfo": {
            "devEui": "ABCDEF1234567890"
        },
        "object": {
            "temperature": round(random.uniform(24.0, 28.0), 1),      # °C
            "ph": round(random.uniform(6.8, 7.8), 2),                 # pH
            "tds": random.randint(100, 400),                          # ppm
            "dissolved_oxygen": round(random.uniform(5.0, 9.0), 2),   # mg/L
            "turbidity": round(random.uniform(5, 90.0), 2),           # NTU
            "depth": round(random.uniform(0.5, 3.0), 2),              # meters
            "pressure": round(random.uniform(1010.3, 1300.0), 2)        # hPa
        }
    }

    client.publish(
        "application/1/device/ABCDEF1234567890/event/up",
        json.dumps(payload)
    )

    print(json.dumps(payload, indent=2))

    time.sleep(60)