
import os
import uuid

from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)

app.config["SECRET_KEY"] = "change-this-secret-key"
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "static", "results")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load your trained YOLO model
model = None

def get_model():
    global model
    if model is None:
        model = YOLO(MODEL_PATH)
    return model

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        if "image" not in request.files:
            return render_template(
                "index.html",
                error="Please select an image."
            )

        file = request.files["image"]

        if file.filename == "":
            return render_template(
                "index.html",
                error="No image selected."
            )

        if not allowed_file(file.filename):
            return render_template(
                "index.html",
                error="Please upload a JPG, JPEG, PNG or WEBP image."
            )

       
        try:
            original_name = secure_filename(file.filename)
            extension = original_name.rsplit(".", 1)[1].lower()

            unique_name = f"{uuid.uuid4().hex}.{extension}"

            upload_path = os.path.join(
                UPLOAD_FOLDER,
                unique_name
            )

            file.save(upload_path)

            # Run object detection
            yolo_model = get_model()

            predictions = yolo_model.predict(
                source=upload_path,
                conf=0.25,
                imgsz=416,
                device="cpu",
                verbose=False
            )

            result = predictions[0]

            # Draw bounding boxes
            plotted_image = result.plot()

            output_image = Image.fromarray(
                plotted_image[:, :, ::-1]
            )

            result_name = f"result_{uuid.uuid4().hex}.jpg"

            result_path = os.path.join(
                RESULT_FOLDER,
                result_name
            )

            output_image.save(result_path, format="JPEG")

            # Extract detection results
            detections = []

            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())

                detections.append({
                    "class_name": yolo_model.names[class_id],
                    "confidence": round(confidence * 100, 2)
                })

            return render_template(
                "result.html",
                original_image=url_for(
                    "static",
                    filename=f"uploads/{unique_name}"
                ),
                result_image=url_for(
                    "static",
                    filename=f"results/{result_name}"
                ),
                detections=detections,
                total=len(detections)
            )

        except Exception as e:
            app.logger.exception("Detection failed")

            return render_template(
                "index.html",
                error=f"Detection failed: {str(e)}"
            )


    return render_template("index.html")
if __name__ == "__main__":
    app.run(debug=True)
