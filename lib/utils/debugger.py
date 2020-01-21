from __future__ import absolute_import, division, print_function

import math

import cv2
import numpy as np


class Debugger(object):
    def __init__(self, ipynb=False, theme='black', 
               num_classes=-1, dataset=None, down_ratio=4):
        self.ipynb = ipynb
        if not self.ipynb:
            import matplotlib.pyplot as plt
            self.plt = plt
        self.imgs = {}
        self.theme = theme
        colors = [(color_list[_]).astype(np.uint8) \
                for _ in range(len(color_list))]
        self.colors = np.array(colors, dtype=np.uint8).reshape(len(colors), 1, 1, 3)
        if self.theme == 'white':
            self.colors = self.colors.reshape(-1)[::-1].reshape(len(colors), 1, 1, 3)
            self.colors = np.clip(self.colors, 0., 0.6 * 255).astype(np.uint8)
        self.dim_scale = 1
        if dataset == 'coco_hp':
            self.names = ['p']
            self.num_class = 1
            self.num_joints = 17
            self.edges = [[0, 1], [0, 2], [1, 3], [2, 4], 
                        [3, 5], [4, 6], [5, 6], 
                        [5, 7], [7, 9], [6, 8], [8, 10], 
                        [5, 11], [6, 12], [11, 12], 
                        [11, 13], [13, 15], [12, 14], [14, 16]]
            self.ec = [(255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255), 
                     (255, 0, 0), (0, 0, 255), (255, 0, 255),
                     (255, 0, 0), (255, 0, 0), (0, 0, 255), (0, 0, 255),
                     (255, 0, 0), (0, 0, 255), (255, 0, 255),
                     (255, 0, 0), (255, 0, 0), (0, 0, 255), (0, 0, 255)]
            self.colors_hp = [(255, 0, 255), (255, 0, 0), (0, 0, 255), 
            (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
            (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
            (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
            (255, 0, 0), (0, 0, 255)]
        elif num_classes == 80 or dataset == 'coco':
            self.names = coco_class_name
        elif num_classes == 20 or dataset == 'pascal':
            self.names = pascal_class_name
        elif dataset == 'gta':
            self.names = gta_class_name
            self.focal_length = 935.3074360871937
            self.W = 1920
            self.H = 1080
            self.dim_scale = 3
        elif dataset == 'viper':
            self.names = gta_class_name
            self.focal_length = 1158
            self.W = 1920
            self.H = 1080
            self.dim_scale = 3
        elif num_classes == 3 or dataset == 'kitti':
            self.names = kitti_class_name
            self.focal_length = 721.5377
            self.W = 1242
            self.H = 375
        num_classes = len(self.names)
        self.down_ratio=down_ratio
        # for bird view
        self.world_size = 64
        self.out_size = 384

    def add_img(self, img, img_id='default', revert_color=False):
        if revert_color:
            img = 255 - img
        self.imgs[img_id] = img.copy()
  
    def add_mask(self, mask, bg, imgId = 'default', trans = 0.8):
        self.imgs[imgId] = (mask.reshape(
            mask.shape[0], mask.shape[1], 1) * 255 * trans + \
            bg * (1 - trans)).astype(np.uint8)
  
    def show_img(self, pause = False, imgId = 'default'):
        cv2.imshow('{}'.format(imgId), self.imgs[imgId])
        if pause:
            cv2.waitKey()
  
    def add_blend_img(self, back, fore, img_id='blend', trans=0.7):
        if self.theme == 'white':
            fore = 255 - fore
        if fore.shape[0] != back.shape[0] or fore.shape[0] != back.shape[1]:
            fore = cv2.resize(fore, (back.shape[1], back.shape[0]))
        if len(fore.shape) == 2:
            fore = fore.reshape(fore.shape[0], fore.shape[1], 1)
        self.imgs[img_id] = (back * (1. - trans) + fore * trans)
        self.imgs[img_id][self.imgs[img_id] > 255] = 255
        self.imgs[img_id][self.imgs[img_id] < 0] = 0
        self.imgs[img_id] = self.imgs[img_id].astype(np.uint8).copy()
  
    def gen_colormap(self, img, output_res=None):
        img = img.copy()
        c, h, w = img.shape[0], img.shape[1], img.shape[2]
        if output_res is None:
            output_res = (h * self.down_ratio, w * self.down_ratio)
        img = img.transpose(1, 2, 0).reshape(h, w, c, 1).astype(np.float32)
        colors = np.array(
          self.colors, dtype=np.float32).reshape(-1, 3)[:c].reshape(1, 1, c, 3)
        if self.theme == 'white':
            colors = 255 - colors
        color_map = (img * colors).max(axis=2).astype(np.uint8)
        color_map = cv2.resize(color_map, (output_res[0], output_res[1]))
        return color_map

    def gen_colormap_hp(self, img, output_res=None):
        c, h, w = img.shape[0], img.shape[1], img.shape[2]
        if output_res is None:
            output_res = (h * self.down_ratio, w * self.down_ratio)
        img = img.transpose(1, 2, 0).reshape(h, w, c, 1).astype(np.float32)
        colors = np.array(
          self.colors_hp, dtype=np.float32).reshape(-1, 3)[:c].reshape(1, 1, c, 3)
        if self.theme == 'white':
            colors = 255 - colors
        color_map = (img * colors).max(axis=2).astype(np.uint8)
        color_map = cv2.resize(color_map, (output_res[0], output_res[1]))
        return color_map


    def add_rect(self, rect1, rect2, c, conf=1, img_id='default'): 
        cv2.rectangle(
          self.imgs[img_id], (rect1[0], rect1[1]), (rect2[0], rect2[1]), c, 2)
        if conf < 1:
            cv2.circle(self.imgs[img_id], (rect1[0], rect1[1]), int(10 * conf), c, 1)
            cv2.circle(self.imgs[img_id], (rect2[0], rect2[1]), int(10 * conf), c, 1)
            cv2.circle(self.imgs[img_id], (rect1[0], rect2[1]), int(10 * conf), c, 1)
            cv2.circle(self.imgs[img_id], (rect2[0], rect1[1]), int(10 * conf), c, 1)

    def add_coco_bbox(self, bbox, cat, conf=1, show_txt=True, img_id='default'): 
        bbox = np.array(bbox, dtype=np.int32)
        # cat = (int(cat) + 1) % 80
        cat = int(cat)
        # print('cat', cat, self.names[cat])
        c = self.colors[cat][0][0].tolist()
        if self.theme == 'white':
            c = (255 - np.array(c)).tolist()
        txt = '{}{:.1f}'.format(self.names[cat], conf)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
        cv2.rectangle(
          self.imgs[img_id], (bbox[0], bbox[1]), (bbox[2], bbox[3]), c, 2)
        if show_txt:
            cv2.rectangle(self.imgs[img_id],
                        (bbox[0], bbox[1] - cat_size[1] - 2),
                        (bbox[0] + cat_size[0], bbox[1] - 2), c, -1)
            cv2.putText(self.imgs[img_id], txt, (bbox[0], bbox[1] - 2), 
                      font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

    def add_coco_hp(self, points, points_prob, img_id='default'): 
        points = np.array(points, dtype=np.int32).reshape(self.num_joints, 2)
        points_prob = np.array(points_prob, dtype=np.float32).reshape(self.num_joints)

        for j in range(self.num_joints):
            if points_prob[j]>0.:
                cv2.circle(self.imgs[img_id],
                          (points[j, 0], points[j, 1]), 2, (255,255,255), -1)
                         
        stickwidth = 2
        cur_canvas = self.imgs[img_id].copy()             
        for j, e in enumerate(self.edges):
            if points_prob[e[0]] > 0. and points_prob[e[1]] > 0.:
                X = [points[e[0], 1], points[e[1], 1]]
                Y = [points[e[0], 0], points[e[1], 0]]
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                polygon = cv2.ellipse2Poly((int(mY),int(mX)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
                cv2.fillConvexPoly(cur_canvas, polygon, (255, 255, 255))
                self.imgs[img_id] = cv2.addWeighted(self.imgs[img_id], 0.8, cur_canvas, 0.2, 0)

    def add_points(self, points, img_id='default'):
        num_classes = len(points)
        # assert num_classes == len(self.colors)
        for i in range(num_classes):
            for j in range(len(points[i])):
                c = self.colors[i, 0, 0]
                cv2.circle(self.imgs[img_id], (points[i][j][0] * self.down_ratio, 
                                               points[i][j][1] * self.down_ratio),
                           5, (255, 255, 255), -1)
                cv2.circle(self.imgs[img_id], (points[i][j][0] * self.down_ratio,
                                               points[i][j][1] * self.down_ratio),
                           3, (int(c[0]), int(c[1]), int(c[2])), -1)

    def add_coco_seg(self, seg, img_id='default'):
        seg = seg > 0
        color = np.array([[np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)]])
        self.imgs[img_id][seg] = self.imgs[img_id][seg]*0.2 + color*0.8
      
    def show_all_imgs(self, pause=False, time=0):
        if not self.ipynb:
            for i, v in self.imgs.items():
                cv2.imshow('{}'.format(i), v)
            if cv2.waitKey(0 if pause else 1) == 27:
                import sys
                sys.exit(0)
        else:
            self.ax = None
            nImgs = len(self.imgs)
            fig=self.plt.figure(figsize=(nImgs * 10,10))
            nCols = nImgs
            nRows = nImgs // nCols
            for i, (k, v) in enumerate(self.imgs.items()):
                fig.add_subplot(1, nImgs, i + 1)
                if len(v.shape) == 3:
                    self.plt.imshow(cv2.cvtColor(v, cv2.COLOR_BGR2RGB))
                else:
                    self.plt.imshow(v)
            self.plt.show()

    def save_img(self, imgId='default', path='./cache/debug/'):
        cv2.imwrite(path + '{}.png'.format(imgId), self.imgs[imgId])
    
    def save_all_imgs(self, path='./cache/debug/', prefix='', genID=False):
        if genID:
            try:
                idx = int(np.loadtxt(path + '/id.txt'))
            except:
                idx = 0
            prefix=idx
            np.savetxt(path + '/id.txt', np.ones(1) * (idx + 1), fmt='%d')
        for i, v in self.imgs.items():
            cv2.imwrite(path + '/{}{}.png'.format(prefix, i), v)

kitti_class_name = [
  'p', 'v', 'b'
]

gta_class_name = [
  'p', 'v'
]

pascal_class_name = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", 
  "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", 
  "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

coco_class_name = [
     'person', 'bicycle', 'car', 'motorcycle', 'airplane',
     'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
     'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
     'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
     'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
     'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
     'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
     'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
     'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
     'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
     'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
     'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
     'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

color_list = np.array(
        [
            1.000, 1.000, 1.000,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.167, 0.000, 0.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            0.000, 0.447, 0.741,
            0.50, 0.5, 0
        ]
    ).astype(np.float32)
color_list = color_list.reshape((-1, 3)) * 255
