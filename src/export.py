from ultralytics import YOLO

# Load a model
model = YOLO("yolov11-experiment/exp2/weights/best.pt")  # load a custom trained model

# Export the model
model.export(format="onnx")