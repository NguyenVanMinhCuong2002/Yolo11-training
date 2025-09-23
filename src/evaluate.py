from ultralytics import YOLO
import os 

dir = "datasets/test/images/"
list_filename = os.listdir(dir)

# Load a model
model = YOLO("yolov11-experiment/exp2/weights/best.pt")  # pretrained YOLO11n model

for i, filename in enumerate(list_filename, 0):
# Run batched inference on a list of images
    path = os.path.join(dir, filename)
    results = model.predict(path, imgsz=256, conf=0.1, iou=0.25)  # return a list of Results objects

    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        result.save(filename=f"predict/{i}.jpg")  # save to disk