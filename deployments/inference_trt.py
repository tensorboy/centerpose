from centernet_tensorrt_engine import CenterNetTensorRTEngine
import cv2
import pickle

config = '../experiments/res_50_512x512_sgd.yaml'
engine = CenterNetTensorRTEngine(weight_version=1)
img = cv2.imread('../images/33887522274_eebd074106_k.jpg')
detections = engine.run(img)
print(detections)
