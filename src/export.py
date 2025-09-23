from ultralytics import YOLO

weight_path = "" # copy best weight to here

# Load a model
model = YOLO(weight_path)  # load a custom trained model

# Export the model
model.export(format="onnx")