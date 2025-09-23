import os
import mlflow
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

# MLflow config
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("yolov11-experiment")

# Load model
model = YOLO("yolo11l.pt")

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
        project="yolov11-experiment",
        name="exp",
        save_period=10   # YOLO sẽ lưu epoch10.pt, epoch20.pt, ...
    )

    # Sau khi train xong, upload thêm best.pt
    # best_model = model.trainer.save_dir / "weights" / "best.pt"
    # if best_model.exists():
    #     mlflow.log_artifact(str(best_model))
