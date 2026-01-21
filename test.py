
from ultralytics import YOLO
import yaml
# 加载自定义模型
model1 = YOLO('ultralytics/cfg/models/v8/yolov8s.yaml',verbose=True)
model2 = YOLO('ultralytics/cfg/models/v8_improved/yolov8s_detect_sa.yaml',verbose=True)
model3 = YOLO('ultralytics/cfg/models/v8_improved/yolov8s-p2_BiFPN.yaml',verbose=True)
# model4 = YOLO('ultralytics/cfg/models/v8_improved/yolov8_DASM.yaml',verbose=True)
model5 = YOLO('ultralytics/cfg/models/v8_improved/yolov8s_AFPN.yaml',verbose=True)

model1.info(verbose=True)
model2.info(verbose=True)
model3.info(verbose=True)
# model4.info(verbose=True)
model5.info(verbose=True)
# test_flops.py
