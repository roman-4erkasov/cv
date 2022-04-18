import base64
import torchvision
import cv2
import torchvision
from io import BytesIO
from flask import Flask
from matplotlib.figure import Figure

from visualizer import Visualizer
from model import load_model, UNet
app = Flask(__name__)
import config
import torch
import numpy as np

# FLASK_APP=app.py flask run

@app.route("/print_filename", methods=['POST','PUT'])
def print_filename():
    """
    curl -X POST -F file=@test.txt http://127.0.0.1:5000/print_filename
    """
    file = request.files['file']
    filename=secure_filename(file.filename)   
    return filename


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


def make_predictions(model, imagePath):
    # set model to evaluation mode
    model.eval()
    # turn off gradient tracking
    with torch.no_grad():
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig = image.copy()
        image = image.astype("float32") / 255.0
        image = cv2.resize(image, (128, 128))
        # orig = image.copy()
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        image = torch.from_numpy(image).to(config.DEVICE)
        # make the prediction, pass the results through the sigmoid
        # function, and convert the result to a NumPy array
        predMask = model(image).squeeze()
        predMask = torch.sigmoid(predMask)
        predMask = predMask.cpu()#.numpy()
        # filter out the weak predictions and convert them to integers
        predMask = (predMask > config.THRESHOLD) * 255
        predMask = predMask.to(torch.bool)
        return orig, image,  predMask



image_raw = cv2.cvtColor(
    cv2.imread(
        # '/home/ubuntu/tgs-salt-identification-challenge/train/images/000e218f21.png'        
        '/home/ubuntu/tgs-salt-identification-challenge/train/images/0061281eea.png'
    ), 
    cv2.COLOR_BGR2RGB
)

image = prepare_image(image_raw)
model = load_model(config.PATH_MODEL)


# @app.route('/upload', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         # check if the post request has the file part
#         if 'file' not in request.files:
#             flash('No file part')
#             return redirect(request.url)
#         file = request.files['file']
#         # If the user does not select a file, the browser submits an
#         # empty file without a filename.
#         if file.filename == '':
#             flash('No selected file')
#             return redirect(request.url)
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#             return redirect(url_for('download_file', name=filename))
#     return '''
#     <!doctype html>
#     <title>Upload new File</title>
#     <h1>Upload new File</h1>
#     <form method=post enctype=multipart/form-data>
#       <input type=file name=file>
#       <input type=submit value=Upload>
#     </form>
#     '''



@app.route("/")
def hello():
    # Generate the figure **without using pyplot**.
    fig = Figure(figsize=(10,10))
    ax = fig.subplots()
    # ax.plot([1, 2])
    # Save it to a temporary buffer.
    img_orig, img_res, mask = make_predictions(
        model, 
        '/home/ubuntu/tgs-salt-identification-challenge/'
        'train/images/0061281eea.png'
    )

    Visualizer.show(
        img=torch.from_numpy(
            cv2.resize(img_orig, (128, 128)).transpose(2, 0, 1)
        ), 
        masks=mask,#.to(torch.uint8)
        ax=ax
    )
    buf = BytesIO()
    fig.savefig(buf, format="png")
    # Embed the result in the html output.
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    print(data)
    return f"<img src='data:image/png;base64,{data}'/>"
