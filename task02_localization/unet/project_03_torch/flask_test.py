import base64
import torchvision
import cv2

from io import BytesIO
from flask import Flask
from matplotlib.figure import Figure

app = Flask(__name__)


# FLASK_APP=web_application_server_sgskip flask run


def prepare_image(image):
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(
                (config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)
            ),
            torchvision.transforms.ToTensor()
        ]
    )
    if transforms is not None:
        image = transforms(image)
    return image

image_raw = cv2.cvtColor(
    cv2.imread(
        '/home/ubuntu/tgs-salt-identification-challenge/train/images/000e218f21.png'        
    ), 
    cv2.COLOR_BGR2RGB
)

image = prepare_image(image_raw)
model = load_model(config.PATH_MODEL)


@app.route("/")
def hello():
    # Generate the figure **without using pyplot**.
    fig = Figure()
    ax = fig.subplots()
    ax.plot([1, 2])
    # Save it to a temporary buffer.
    buf = BytesIO()
    fig.savefig(buf, format="png")
    # Embed the result in the html output.
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"<img src='data:image/png;base64,{data}'/>"
