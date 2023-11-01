from ultralytics import YOLO


if __name__ == '__main__':

    model = YOLO("ultralytics/cfg/models/v8/yolov8.yaml").train(**{'cfg':'ultralytics/cfg/default.yaml'})