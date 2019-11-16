import os
import json
import base64
import logging

import tensorrt as trt
from google.cloud import storage


logger = logging.getLogger(__name__)
TRT_LOGGER = trt.Logger()  # required by TensorRT

_SAVE_DIR = '.nn-weights'
_AVAILABLE_MODEL_TYPES = ('mask_rcnn', 'yolo')
_AVAILABLE_ENGINE_TYPES = ('yolo', 'openpose', 'centernet')


def build_credentials_from_env():
    """Create credential file from environmental args on the fly"""
    credential_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if credential_path is None:
        logger.error("Missing GCP credential information. Please check GOOGLE_APPLICATION_CREDENTIALS.")
        raise RuntimeError
    if not os.path.isfile(credential_path):
        private_key = os.getenv("PRIVATE_GCP_BASE64_KEY")
        if private_key is None:
            logger.error("Cannot build GCP credentials since missing private key Please check PRIVATE_GCP_BASE64_KEY.")
            raise RuntimeError
        logger.info("Creating credentials from private key for GCP and dumping it to {}...".format(
            credential_path))
        key_data = json.loads(base64.b64decode(private_key))
        with open(credential_path, 'w') as f:
            json.dump(key_data, f)


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


def load_engine(onnx_file_path, engine_file_path, precision, max_batch_size, cache_file=None):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    print(onnx_file_path)
    build_engine(onnx_file_path, engine_file_path, precision, max_batch_size, cache_file=cache_file)

    logger.info("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def fetch_data_from_gcp(bucket_name, file_path, dest_path):
    """Download data from Google Cloud Platform and save it locally
    """
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(file_path)
    blob.download_to_filename(dest_path)


def load_nn_weights(model_type, version, map_location=None):
    """Loads model weights for given model type.

    If the object is already present at `_WEIGHTS_DIR`, it's loaded and returned.
    Otherwise it will download model weights from AiFi cloud storage.

    Args:
        model_type (string): model type. All valid types are shown in `_AVAILABLE_MODEL_TYPES`.
        map_location (optional): showing which device should weights be remapped to (see torch.load)

    Example:
        >>> state_dict = load_nn_weights(model_type='yolo', map_location='cpu')
    """
    import torch

    assert model_type in _AVAILABLE_MODEL_TYPES, "model_type could only be one of {}".format(
        _AVAILABLE_MODEL_TYPES)

    weights_dir = os.path.join(_SAVE_DIR, model_type)
    if not os.path.isdir(weights_dir):
        os.makedirs(weights_dir)
    weights_file = os.path.join(weights_dir, 'ckpt{}.pth'.format(version))
    if not os.path.isfile(weights_file):
        logger.info('Downloading {} weights version {} from AiFi Google Cloud Storage...'.format(
            model_type, version))
        fetch_data_from_gcp('nn-weights', '{}/ckpt{}.pth'.format(model_type, version), weights_file)
    return torch.load(weights_file, map_location=map_location)


def create_trt_context(model_type, version, precision='fp32', max_batch_size=1, save_only=False):
    """Loads onnx file for given model type, build tensorrt engine, and create runtime context

    If the onnx file is already present at `_SAVE_DIR`, it will be loaded. Otherwise it will
    download onnx file from AiFi cloud storage.

    Args:
        model_type (string): model type. All valid types are shown in `_AVAILABLE_ENGINE_TYPES`.

    Example:
        >>> context = create_trt_context(model_type='yolo')
    """

    assert model_type in _AVAILABLE_ENGINE_TYPES, "model_type could only be one of {} but found {}".format(
        _AVAILABLE_ENGINE_TYPES, model_type)

    onnx_dir = os.path.join(_SAVE_DIR, model_type)
    if not os.path.isdir(onnx_dir):
        os.makedirs(onnx_dir)
    onnx_file = os.path.join(onnx_dir, 'ckpt{}.onnx'.format(version))
    if not os.path.isfile(onnx_file):
        logger.info('Downloading {} onnx file version {} from AiFi Google Cloud Storage...'.format(
            model_type, version))
        fetch_data_from_gcp('nn-weights', '{}/ckpt{}.onnx'.format(model_type, version), onnx_file)

    engine_file = os.path.join(onnx_dir, 'engine{}_{}_bs{}.trt'.format(version, precision, max_batch_size))

    cache_file = None
    # if precision is int8 and no engine found, load calibration cache from cloud.
    if precision == 'int8' and not os.path.isfile(engine_file):
        cache_file = os.path.join(onnx_dir, 'calibration{}.cache'.format(version))
        fetch_data_from_gcp('nn-weights', '{}/calibration{}.cache'.format(model_type, version), cache_file)

    if save_only:
        build_engine(onnx_file, engine_file, precision=precision, max_batch_size=max_batch_size, cache_file=cache_file)
    else:
        engine = load_engine(
            onnx_file, engine_file, precision=precision, max_batch_size=max_batch_size, cache_file=cache_file)
        return engine.create_execution_context()
