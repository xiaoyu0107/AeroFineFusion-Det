import warnings

from sympy import false

warnings.filterwarnings('ignore')
from ultralytics import RTDETR, YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/yolov8s.yaml')
    # model = RTDETR('/data/experiments/CXY/ultralytics-main/ultralytics/cfg/models/rt-detr/rtdetr-resnet50.yaml')
    model.train(data='/data/experiments/CXY/ultralytics-main/ultralytics/cfg/datasets/VisDrone.yaml',
                cache=False,
                imgsz=640,
                epochs=250,
                batch=8,
                close_mosaic=10,
                workers=4,
                patience=300,
                optimizer='SGD',  # using SGDs
                project='runs/visdrone',
                name='train_yolov8s',
                resume=False,
                device='cuda:0',
                amp=True,
                )