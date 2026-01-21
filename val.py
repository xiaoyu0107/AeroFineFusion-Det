from ultralytics import YOLO
import os

# 模型路径
model_path1 = '/data/experiments/CXY/ultralytics-main/runs/visdrone/train_yolov8s+p2-p5+QA3/weights/best.pt'  # 替换为你的模型路径
model_path2 = 'runs/visdrone/train_yolov8s-p2_BiFPN/weights/best.pt'
# # 自定义保存路径
# save_dir = 'runs/visdrone/val_yolov8s_250e'  # 可更改为你希望的目录
# os.makedirs(save_dir, exist_ok=True)

# 加载模型
model = YOLO(model_path2)

visdrone='/data/experiments/CXY/ultralytics-main/ultralytics/cfg/datasets/VisDrone.yaml'
seadronesee='/data/experiments/CXY/ultralytics-main/ultralytics/cfg/datasets/seadronesee.yaml'
# 运行验证
metrics = model.val(
    data=visdrone,       # 替换为你的数据配置路径
    split='val',            # 使用验证集
    save_json=True,
    conf=0.001,             # 低阈值保留更多预测
    iou=0.7,
    max_det=300,
    half=False,
    dnn=False,
    plots=True,             # 保存混淆矩阵、PR曲线等
    save=True,              # 保存预测图片
    save_txt=False,         # 是否保存标签txt
    save_conf=False,        # 是否保存置信度
    project='runs/visdrone2',     # 基础保存目录
    name='val_yolov8s-p2_BiFPN', # 子目录名
    device=1,
)
print(metrics.box.map, metrics.box.map50, metrics.box.map75)

#
# print(f"验证完成，结果保存在：{save_dir}")
