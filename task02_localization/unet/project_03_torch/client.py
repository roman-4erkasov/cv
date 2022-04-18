import requests

path = "'/home/ubuntu/tgs-salt-identification-challenge/train/images/000e218f21.png'"
path = '<PATH/TO/.jpg/FILE>/cat.jpg'
with open(path, "rb"):
    resp = requests.post(
        "http://localhost:5000/predict",
        files={"file": fp}
    )
