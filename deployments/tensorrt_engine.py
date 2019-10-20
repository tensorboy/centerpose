import logging

import numpy as np
import pycuda.driver as cuda

from .utils.config_parse import merge_cfg_from_file
from .utils.trt_utils import allocate_buffers
from .utils.gcp_utils import build_credentials_from_env


logger = logging.getLogger(__name__)

build_credentials_from_env()


class TensorRTEngine:
    def __init__(self, config_file, weight_version, precision, max_batch_size):
        """General tensorrt engine object.

        It wraps up neural network model with its specific preprocess and postprocess, and provides a `run`
        method for end-to-end inference. Model should come from onnx file or tensorrt engine file.

        Args:
            config_file: config file which can be used to overwrite some default params
            precision: tensorrt engine precision
            max_batch_size: tensorrt engine maximum input batch size
        """
        cuda.init()
        dev = cuda.Device(0)
        self.cuda_ctx = dev.make_context()

        self._initialize_config(config_file)
        self.precision = precision
        self.weight_version = weight_version
        self.max_batch_size = max_batch_size

        self._initialize_context()
        self._initialize_buffers()

    def __del__(self):
        """clean GPU context upon destruction"""
        logger.info("Detector: id={}, cleanning cuda context...")
        self.cuda_ctx.pop()
        del self.cuda_ctx

    def _initialize_config(self, config_file):
        """Initializ config file for model/preprocess/postprocess params.
        Configs will be saved in `self.cfg` as AttrDict.

        Args:
            config_file (str): path to yaml config file
            cfg (AttrDict): Default config params in config.py
        """
        cfg = self._load_config()
        if config_file is not None:
            merge_cfg_from_file(config_file, cfg)
        self.cfg = cfg

    def _initialize_buffers(self):
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(
            self.context.engine)

    def _initialize_context(self):
        """Initialize tensorrt execution context.
        Engine uses base class `tensorrt.IExecutionContext` and is saved in self.context
        """
        raise NotImplementedError

    def _load_config(self):
        """Load default config parameters. This function should return an AttrDict object
        """
        raise NotImplementedError

    def _preprocess(self, imgs):
        """Image preprocessing
        Args:
            imgs (np.ndarray or list of ndarrays): img input with HWC format
        """
        raise NotImplementedError

    def forward(self, batch_size):
        """Tensorrt engine execution function
        Args:
            batch_size (int): input batch size
        """
        if batch_size > self.max_batch_size:
            logger.error("Batch size ({}) is larger than maximum batch size ({}) of this engine.".format(
                batch_size, self.max_batch_size))
            raise RuntimeError
        [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]
        self.context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
        [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
        self.stream.synchronize()

    def _postprocess(self, batch_size):
        """Postprocessing
        Args:
            batch_size (int): input batch size
        """
        raise NotImplementedError

    def run(self, imgs):
        """Engine callable method to run preprocessing, inference and postprocessing in end-to-end manner.

        Args:
            imgs (np.ndarray or list of ndarrays): numpy image array with shape (H, W, 3), or list of image arrays

        Returns:
            predictions: engine output
        """

        if isinstance(imgs, np.ndarray):
            batch_size = 1
        else:
            batch_size = len(imgs)

        self._preprocess(imgs)
        self.forward(batch_size)
        predictions = self._postprocess(batch_size)

        return predictions
