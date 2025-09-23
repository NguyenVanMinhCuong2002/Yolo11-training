import os
import mlflow
from ultralytics import YOLO
from dotenv import load_dotenv
import os


mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
model_name = os.getenv("YOLO_MODEL_NAME")

load_dotenv()

# MLflow config
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment(f"{model_name}-experiment")

# Load model
model = YOLO(model_name)

# Custom callback: upload model mỗi 10 epoch
def log_every_10_epochs(trainer):
    epoch = trainer.epoch + 1   # epoch bắt đầu từ 0
    if epoch % 10 == 0:
        weights_dir = trainer.save_dir / "weights"
        ckpt = weights_dir / f"epoch{epoch}.pt"
        if ckpt.exists():
            print(f"[MLflow] Upload checkpoint: {ckpt}")
            mlflow.log_artifact(str(ckpt))

# Đăng ký callback
model.add_callback("on_fit_epoch_end", log_every_10_epochs)

with mlflow.start_run():
    results = model.train(
        
        data="datasets/data.yaml",
        epochs=350,
        imgsz=256,
        batch=64,
        project=f"{model_name}-experiment",
        name="exp",
        save_period=10   # YOLO sẽ lưu epoch10.pt, epoch20.pt, ...
    )

