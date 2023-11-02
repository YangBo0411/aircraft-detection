from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor

from ultralytics import RTDETR, YOLO
from ultralytics.data.build import load_inference_source
from ultralytics.utils import LINUX, ONLINE, ROOT, SETTINGS


CFG = 'ultralytics/cfg/models/v8/yolov8-EVC.yaml'
SOURCE = ROOT / 'assets/bus.jpg'

def test_model_forward():
    model = YOLO(CFG)
    model(SOURCE)