from centernet_tensorrt_engine import CenterNetTensorRTEngine
import os
import cv2
import math
import time
import torch
import numpy as np
import pickle
import tensorrt as trt
import logging
from face.centerface import CenterFace
from face.prnet import PRN
from face.utils.estimate_pose import estimate_pose
from torchvision import transforms
from face.utils.cv_plot import plot_kpt, plot_vertices, plot_pose_box

logger = logging.getLogger(__name__)
TRT_LOGGER = trt.Logger()  # required by TensorRT 
      

colors = [tuple(np.random.choice(np.arange(256).astype(np.int32), size=3)) for i in range(100)]

def kp_connections(keypoints):
    kp_lines = [
        [keypoints.index('nose'), keypoints.index('left_eye')],
        [keypoints.index('left_eye'), keypoints.index('left_ear')],
        [keypoints.index('nose'), keypoints.index('right_eye')],
        [keypoints.index('right_eye'), keypoints.index('right_ear')],                 
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
        [keypoints.index('right_shoulder'), keypoints.index('right_hip')],
        [keypoints.index('right_hip'), keypoints.index('right_knee')],                 
        [keypoints.index('right_knee'), keypoints.index('right_ankle')], 
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        [keypoints.index('left_shoulder'), keypoints.index('left_hip')],
        [keypoints.index('left_hip'), keypoints.index('left_knee')],  
        [keypoints.index('left_knee'), keypoints.index('left_ankle')],           
    ]
    return kp_lines


def get_keypoints():
    """Get the COCO keypoints and their left/right flip coorespondence map."""
    keypoints = [
    'nose',            # 1
    'left_eye',        # 2
    'right_eye',       # 3
    'left_ear',        # 4
    'right_ear',       # 5
    'left_shoulder',   # 6
    'right_shoulder',  # 7
    'left_elbow',      # 8
    'right_elbow',     # 9
    'left_wrist',      # 10
    'right_wrist',     # 11
    'left_hip',        # 12
    'right_hip',       # 13
    'left_knee',       # 14
    'right_knee',      # 15
    'left_ankle',      # 16
    'right_ankle',     # 17
    ]

    return keypoints


_kp_connections = kp_connections(get_keypoints())
      
def build_engine(onnx_file_path, engine_file_path, precision, max_batch_size, cache_file=None):
    """Builds a new TensorRT engine and saves it, if no engine presents"""

    if os.path.exists(engine_file_path):
        logger.info('{} TensorRT engine already exists. Skip building engine...'.format(precision))
        return

    logger.info('Building {} TensorRT engine from onnx file...'.format(precision))
    with trt.Builder(TRT_LOGGER) as b, b.create_network() as n, trt.OnnxParser(n, TRT_LOGGER) as p:
        b.max_workspace_size = 1 << 30  # 1GB
        b.max_batch_size = max_batch_size
        if precision == 'fp16':
            b.fp16_mode = True
        elif precision == 'int8':
            from ..calibrator import Calibrator
            b.int8_mode = True
            b.int8_calibrator = Calibrator(cache_file=cache_file)
        elif precision == 'fp32':
            pass
        else:
            logger.error('Engine precision not supported: {}'.format(precision))
            raise NotImplementedError
        # Parse model file
        with open(onnx_file_path, 'rb') as model:
            p.parse(model.read())
        if p.num_errors:
            logger.error('Parsing onnx file found {} errors.'.format(p.num_errors))
        engine = b.build_cuda_engine(n)
        print(engine_file_path)
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())

def add_coco_bbox(image, bbox, cat, conf=1, color=None): 
    cat = int(cat)
    txt = '{}{:.1f}'.format('person', conf)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (int(color[0]), int(color[1]), int(color[2])), 2)
    cv2.putText(image, txt, (bbox[0], bbox[1] - 2), 
              font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

def add_coco_hp(image, points, color): 
    for j in range(17):
        cv2.circle(image,
                 (points[j, 0], points[j, 1]), 2, (int(color[0]), int(color[1]), int(color[2])), -1)
                 
    stickwidth = 2
    cur_canvas = image.copy()             
    for j, e in enumerate(_kp_connections):
        if points[e].min() > 0:
            X = [points[e[0], 1], points[e[1], 1]]
            Y = [points[e[0], 0], points[e[1], 0]]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY),int(mX)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, (int(color[0]), int(color[1]), int(color[2])))
            image = cv2.addWeighted(image, 0.5, cur_canvas, 0.5, 0)

    return image
    
build_engine('model/resnet50.onnx', 'model/resnet50.trt', 'fp32', 1)            
config = '../experiments/res_50_512x512.yaml'
body_engine = CenterNetTensorRTEngine(weight_file='model/resnet50.trt', config_file=config)

face_model_path = '/home/tensorboy/Documents/centerface/models/onnx/centerface.onnx'
face_engine = CenterFace(model_path = face_model_path, landmarks = True)

face_3d_model_path = '/home/tensorboy/data/demo/weights/prnet.pth' 
face_3d_model = PRN(face_3d_model_path, '/home/tensorboy/centerpose/demo/face')

image = cv2.imread('../images/image1.jpg')
rgb_img =  cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
detections = body_engine.run(image)[1]

face_engine.transform(image.shape[0], image.shape[1])
face_dets, lms = face_engine(image, threshold=0.35)
print(face_dets.shape)   

for i, bbox in enumerate(detections):
    if bbox[4] > 0.3:
        color = colors[i]
        bbox = np.array(bbox, dtype=np.int32)
        body_bbox = bbox[:4]
        body_prob = bbox[4]
        body_pose = bbox[5:39]
        keypoints = np.array(body_pose, dtype=np.int32).reshape(17, 2)
        center_of_the_face = np.mean(keypoints[:7,:], axis=0)
                 
        face_min_dis = np.argmin(np.sum(((face_dets[:,2:4]+face_dets[:,:2])/2.-center_of_the_face)**2,axis=1))
            
        face_bbox = face_dets[face_min_dis][:4]
        face_prob = face_dets[face_min_dis][4]
        
        face_image = rgb_img[int(face_bbox[1]): int(face_bbox[3]), int(face_bbox[0]): int(face_bbox[2])]
        # the core: regress position map
        #cv2.imwrite('face%d.jpg'%i, face_image)
        [h, w, c] = face_image.shape          
          
        box = np.array([0, face_image.shape[1]-1, 0, face_image.shape[0]-1]) # cropped with bounding box
        pos = face_3d_model.process(face_image, box)    
        
        vertices = face_3d_model.get_vertices(pos)
        save_vertices = vertices.copy()
        save_vertices[:, 1] = h - 1 - save_vertices[:, 1]
        
        kpt = face_3d_model.get_landmarks(pos)
        camera_matrix, pose = estimate_pose(vertices)
                    
        bgr_face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        image_pose = plot_pose_box(bgr_face_image, camera_matrix, kpt)
        sparse_face =  plot_kpt(bgr_face_image, kpt)

        dense_face = plot_vertices(bgr_face_image, vertices)
        image[int(face_bbox[1]): int(face_bbox[3]), int(face_bbox[0]): int(face_bbox[2])] = cv2.resize(image_pose, (w,h))
          
        #cv2.imshow('image pose', image_pose)
        #cv2.imshow('sparse alignment', sparse_face)
        #cv2.imshow('dense alignment', dense_face)
        #cv2.waitKey(0)
        cv2.rectangle(image, (int(face_bbox[0]), int(face_bbox[1])), (int(face_bbox[2]), int(face_bbox[3])), (int(color[0]), int(color[1]), int(color[2])), 3)
        add_coco_bbox(image, body_bbox, 0, body_prob, color)
        image = add_coco_hp(image, keypoints, color)
cv2.imwrite('result.png', image)


#image_dir = '/data/coco/images/val2017'

#all_times = []
#for image_path in os.listdir(image_dir):
#    image = cv2.imread('../images/33823288584_1d21cf0a26_k.jpg')
#    tic = time.time()
#    detections = engine.run(image)[1]
#    toc = time.time()
#    all_times.append(toc-tic)
#    print('mean times', np.mean(all_times))

