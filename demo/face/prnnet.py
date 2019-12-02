# -*- coding: utf-8 -*-

"""
    @date: 2019.07.19
    @author: samuel ko
    @func: same function as api.py in original PRNet Repo.
"""
import torch
import numpy as np
from model.resfcn256 import ResFCN256


class PRN:
    '''
        <Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network>

        This class serves as the wrapper of PRNet.
    '''

    def __init__(self, model_dir, **kwargs):
        # resolution of input and output image size.
        self.resolution_inp = kwargs.get("resolution_inp") or 256
        self.resolution_op = kwargs.get("resolution_op") or 256
        self.channel = kwargs.get("channel") or 3
        self.size = kwargs.get("size") or 16

        self.uv_kpt_ind_path = kwargs.get("uv_kpt_path") or "utils/uv_data/uv_kpt_ind.txt"
        self.face_ind_path = kwargs.get("face_ind_path") or "utils/uv_data/face_ind.txt"
        self.triangles_path = kwargs.get("triangles_path") or "utils/uv_data/triangles.txt"

        # 1) load model.
        self.pos_predictor = ResFCN256()
        state = torch.load(model_dir)
        self.pos_predictor.load_state_dict(state['prnet'])
        self.pos_predictor.eval()  # inference stage only.
        if torch.cuda.device_count() > 0:
            self.pos_predictor = self.pos_predictor.to("cuda")

        # 2) load uv_file.
        self.uv_kpt_ind = np.loadtxt(self.uv_kpt_ind_path).astype(np.int32)  # 2 x 68 get kpt
        self.face_ind = np.loadtxt(self.face_ind_path).astype(np.int32)  # get valid vertices in the pos map
        self.triangles = np.loadtxt(self.triangles_path).astype(np.int32)  # ntri x 3

        self.uv_coords = self.generate_uv_coords()

    def net_forward(self, image):
        ''' The core of out method: regress the position map of a given image.
        Args:
            image: (3, 256, 256) array. value range: 0~1
        Returns:
            pos: the 3D position map. (3, 256, 256) array.
        '''
        return self.pos_predictor(image)

    def generate_uv_coords(self):
        resolution = self.resolution_op
        uv_coords = np.meshgrid(range(resolution), range(resolution))
        uv_coords = np.transpose(np.array(uv_coords), [1, 2, 0])
        uv_coords = np.reshape(uv_coords, [resolution ** 2, -1])
        uv_coords = uv_coords[self.face_ind, :]
        uv_coords = np.hstack((uv_coords[:, :2], np.zeros([uv_coords.shape[0], 1])))
        return uv_coords

    def get_landmarks(self, pos):
        '''
        Notice: original tensorflow version shape is [256, 256, 3] (H, W, C)
                where our pytorch shape is [3, 256, 256] (C, H, W).

        Args:
            pos: the 3D position map. shape = (256, 256, 3).
        Returns:
            kpt: 68 3D landmarks. shape = (68, 3).
        '''
        kpt = pos[self.uv_kpt_ind[1, :], self.uv_kpt_ind[0, :], :]
        return kpt

    def get_vertices(self, pos):
        '''
        Args:
            pos: the 3D position map. shape = (3, 256, 256).
        Returns:
            vertices: the vertices(point cloud). shape = (num of points, 3). n is about 40K here.
        '''
        all_vertices = np.reshape(pos, [self.resolution_op ** 2, -1])
        vertices = all_vertices[self.face_ind, :]

        return vertices
