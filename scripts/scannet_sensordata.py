# Reader for ScanNet .sens files.
#
# Origin: SensorData.py of the ScanNet toolkit (SensReader/python,
# https://github.com/ScanNet/ScanNet), Copyright 2017 Angela Dai, Angel X. Chang,
# Manolis Savva, Maciej Halber, Thomas Funkhouser, Matthias Niessner; MIT License
# (scripts/LICENSE-SCANNET). The Python 3 port comes from Panoptic Lifting's copy of
# that file (dataset/preprocessing/sens_reader/SensorData.py in
# https://github.com/nihalsid/panoptic-lifting): the print statements are function
# calls and the frame and sensor-name bytes are joined as b''.join instead of the
# Python 2 ''.join. Our own changes on top of that copy: the frame bytes are read
# directly (file_handle.read) instead of b''.join over struct-unpacked characters,
# and the export_* methods, which preprocess_scannetv2_s20.py does not use, were
# removed. preprocess_scannetv2_s20.py uses SensorData(...), .frames,
# .color_compression_type, .depth_compression_type, .depth_height, .depth_width,
# the intrinsic/extrinsic matrices and the frames' camera_to_world and
# decompress_* methods. Needs numpy and imageio.

import struct
import numpy as np
import zlib
import imageio

COMPRESSION_TYPE_COLOR = {-1:'unknown', 0:'raw', 1:'png', 2:'jpeg'}
COMPRESSION_TYPE_DEPTH = {-1:'unknown', 0:'raw_ushort', 1:'zlib_ushort', 2:'occi_ushort'}

class RGBDFrame():

  def load(self, file_handle):
    self.camera_to_world = np.asarray(struct.unpack('f'*16, file_handle.read(16*4)), dtype=np.float32).reshape(4, 4)
    self.timestamp_color = struct.unpack('Q', file_handle.read(8))[0]
    self.timestamp_depth = struct.unpack('Q', file_handle.read(8))[0]
    self.color_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
    self.depth_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
    # Direct read is much faster than struct.unpack('c'*N, ...) + b''.join(...)
    self.color_data = file_handle.read(self.color_size_bytes)
    self.depth_data = file_handle.read(self.depth_size_bytes)


  def decompress_depth(self, compression_type):
    if compression_type == 'zlib_ushort':
       return self.decompress_depth_zlib()
    else:
       raise


  def decompress_depth_zlib(self):
    return zlib.decompress(self.depth_data)


  def decompress_color(self, compression_type):
    if compression_type == 'jpeg':
       return self.decompress_color_jpeg()
    else:
       raise


  def decompress_color_jpeg(self):
    return imageio.imread(self.color_data)


class SensorData:

  def __init__(self, filename):
    self.version = 4
    self.load(filename)


  def load(self, filename):
    with open(filename, 'rb') as f:
      version = struct.unpack('I', f.read(4))[0]
      assert self.version == version
      strlen = struct.unpack('Q', f.read(8))[0]
      self.sensor_name = b''.join(struct.unpack('c'*strlen, f.read(strlen)))
      self.intrinsic_color = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
      self.extrinsic_color = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
      self.intrinsic_depth = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
      self.extrinsic_depth = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
      self.color_compression_type = COMPRESSION_TYPE_COLOR[struct.unpack('i', f.read(4))[0]]
      self.depth_compression_type = COMPRESSION_TYPE_DEPTH[struct.unpack('i', f.read(4))[0]]
      self.color_width = struct.unpack('I', f.read(4))[0]
      self.color_height =  struct.unpack('I', f.read(4))[0]
      self.depth_width = struct.unpack('I', f.read(4))[0]
      self.depth_height =  struct.unpack('I', f.read(4))[0]
      self.depth_shift =  struct.unpack('f', f.read(4))[0]
      num_frames =  struct.unpack('Q', f.read(8))[0]
      self.frames = []
      for i in range(num_frames):
        frame = RGBDFrame()
        frame.load(f)
        self.frames.append(frame)
