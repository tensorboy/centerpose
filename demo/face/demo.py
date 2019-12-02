import cv2
import scipy.io as sio
import os
from centerface import CenterFace


def test_image(image_path, model_path):
    frame = cv2.imread(image_path)
    h, w = frame.shape[:2]
    landmarks = True
    centerface = CenterFace(model_path=model_path, landmarks=landmarks)
    centerface.transform(h, w)
    if landmarks:
        dets, lms = centerface(frame, threshold=0.35)
    else:
        dets = centerface(frame, threshold=0.35)

    for det in dets:
        boxes, score = det[:4], det[4]
        cv2.rectangle(frame, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (2, 255, 0), 1)
    if landmarks:
        for lm in lms:
            cv2.circle(frame, (int(lm[0]), int(lm[1])), 2, (0, 0, 255), -1)
            cv2.circle(frame, (int(lm[2]), int(lm[3])), 2, (0, 0, 255), -1)
            cv2.circle(frame, (int(lm[4]), int(lm[5])), 2, (0, 0, 255), -1)
            cv2.circle(frame, (int(lm[6]), int(lm[7])), 2, (0, 0, 255), -1)
            cv2.circle(frame, (int(lm[8]), int(lm[9])), 2, (0, 0, 255), -1)
    cv2.imshow('out', frame)
    cv2.waitKey(0)



if __name__ == '__main__':
    image_path = '/home/tensorboy/centerpose/images/image1.jpg'
    model_path = '/home/tensorboy/CenterFace/models/onnx/centerface.onnx'
    test_image(image_path, model_path)
