from ultralytics import YOLO

# Load a model
model = YOLO("best.pt")

#Export the model to torchscript format
model.export(format="torchscript")

print("Model exported to torchscript format")