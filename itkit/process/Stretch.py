import os, pdb, math, logging
from typing_extensions import Literal

import cv2
import torch
import numpy as np
from scipy.ndimage import map_coordinates
from torch import Tensor
from mmcv.transforms import BaseTransform
from mmengine.logging import print_log
from mmseg.datasets.transforms import PackSegInputs as _PackSegInputs



def rectangular_to_polar(x, y, center_x, center_y):
    """
    Rectangular coordinates start from 0.
    Standard rectangular coordinate input: x, y
    Rectangular coordinates of the pole: center_x, center_y

    radius: radial distance
    angle: polar angle in radians
    """
    # Use numpy to compute radius
    radius = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    # Use scipy to compute angle
    angle = np.arctan2(y - center_y, x - center_x)
    
    return radius, angle


def polar_to_rectangular(radius, angle, center_x, center_y):
    """
    Rectangular coordinates start from 0.
    radius: radial distance
    angle: polar angle in radians
    center_x, center_y: pole position in rectangular coordinates

    x, y: rectangular coordinates
    """
    x = center_x + radius * math.cos(angle)
    y = center_y + radius * math.sin(angle)
    
    return x, y


class RadialStretch:
    def __init__(self,
                 CornerFactor=1, 
                 GlobalFactor=1, 
                 in_array_shape:tuple=None,
                 direction:str="out", 
                 mmseg_stretch_seg_map:bool=True,
                 stretch_num_workers:int=8):
        # Controls corner stretch intensity
        # Controls global scaling intensity
        # Input matrix shape
        # Stretch direction, 'out' means stretch outward
        # Whether to stretch segmentation label map
        # Worker count for internal multiprocessing stretch
        assert CornerFactor>=0 and GlobalFactor>=1, "[Dataset] Projection Map Init Error: CornerFactor must >=0, GlobalFactor must >=1"
        assert in_array_shape[0]==in_array_shape[1], "[Dataset] Projection Map Init Error: input image must be square"
        assert direction in ['out', 'in'], "[Dataset] Stretch Direction can only be out or in. Out mean Stretch out to a square. In is its reverse operation"
        
        self.CornerFactor = CornerFactor            # Corner stretch intensity
        self.GlobalFactor = GlobalFactor            # Global scaling intensity
        self.in_array_shape = in_array_shape        # Input matrix shape
        self.direction = direction                  # Stretch direction; 'out' means outward
        self.mmseg_stretch_seg_map = mmseg_stretch_seg_map      # Whether to stretch segmentation label map
        self.stretch_num_workers = stretch_num_workers # Worker count for internal multiprocessing
        self._cache_map()

    def _cache_map(self):
        print_log("[Dataset] Caching stretch mapping matrix", "current", logging.INFO)
        # Output matrix: each point has a mapping coordinate
        map_height, map_width = self.in_array_shape
        self.proj_map = np.zeros(shape=(map_height, map_width, 2), dtype=np.uint16)
        # Traverse each pixel to generate mapping matrix
        for y in range(map_height):
            for x in range(map_width):
                self.proj_map[y, x] = self.CoordinateMapping(y, x)
        self.proj_map.setflags(write=False) # Lock mapping matrix
        print_log("[Dataset] Stretch mapping matrix cached", "current", logging.INFO)

    # Calculate the stretch factor for this polar angle, only applicable to square input and output.
    # Input is the polar angle of the current mapping matrix, in radians.
    # Output the radial stretch factor of the mapping matrix at this polar angle, 
    # which should be multiplied with the current mapping matrix's radius externally to get the source radius.
    def stretch_factor(self, map_radians):
        # Input is in radians
        map_radians = abs(map_radians) % (math.pi/2) # Period 90°, symmetric about Y-axis

        # Deprecated: Linear angle mapping with discontinuity points
        # if angle > 45:
        #     angle = 90 - angle  # Central symmetry within period

        # Progressive Cos angle mapping, centrally symmetric within period, but symmetry mode changes.
        # Establish Cartesian coordinate system, X-axis and above are valid, X-axis represents source angle, 
        # Y-axis represents mapped angle (output to deformation parameter calculation)
        # 
        angle = (math.pi/8) * (1 - math.cos(4*map_radians))

        # Deformation parameter
        radial_factor = 1 / (math.cos(angle)**self.CornerFactor)
        if self.direction == 'out':     # Image stretch direction is outward
            radial_factor = 1 / radial_factor
            global_factor = 1 / self.GlobalFactor
        elif self.direction == 'in':    # Image stretch direction is inward
            radial_factor = radial_factor
            global_factor = self.GlobalFactor
        # Final scaling parameter = deformation in this direction * global scaling
        factor = radial_factor * global_factor
        
        return factor

    # 输入处理后矩阵索引，返回源矩阵索引
    def CoordinateMapping(self, map_Y, map_X):
        # Both X and Y start from 0
        # Image origin is top-left; invert Y so origin becomes bottom-left
        true_map_y = self.in_array_shape[0] - map_Y
        # Mapping matrix pole defaults to its center
        map_center_y, map_center_x = self.in_array_shape[0]/2, self.in_array_shape[1]/2
        # Input image pole defaults to its center
        source_center_y, source_center_x = self.in_array_shape[0]/2, self.in_array_shape[1]/2
        # Obtain polar coordinates in mapping matrix
        radius, angle = rectangular_to_polar(map_X, true_map_y, map_center_y, map_center_x)
        # Compute stretch factor for this polar angle
        stretch_factor_of_this_angle = self.stretch_factor(angle)
        # Convert back to rectangular to find source coordinates
        source_x, true_source_y = polar_to_rectangular(radius*stretch_factor_of_this_angle, angle, source_center_x, source_center_y)
        # Round and clip coordinates
        source_x, true_source_y = np.clip(round(source_x), 1, self.in_array_shape[1]), np.clip(round(true_source_y), 1, self.in_array_shape[0])
        # Restore Y after inversion; also restore to index range
        source_y = self.in_array_shape[0] - true_source_y
        # Restore X to index range
        source_x -= 1
        return source_y, source_x

    # 单输入执行拉伸
    def stretch(self, image_matrix, type=Literal['img', 'label'], pad_val=None, seg_pad_val=None):
        # 当参数等效为无拉伸时，直接返回
        if self.CornerFactor==0 and self.GlobalFactor==1:
            return image_matrix
        
        out_shape = self.proj_map.shape[:-1]
        # numpy映射比tensor快一个数量级以上
        # 创建与输入矩阵相同大小的零矩阵
        if isinstance(image_matrix, Tensor):
            if image_matrix.dtype == torch.uint8:
                stretched_matrix = np.zeros(out_shape, dtype=np.uint8)
            else:
                stretched_matrix = np.zeros(out_shape, dtype=np.float32)
        elif isinstance(image_matrix, np.ndarray):
            stretched_matrix = np.zeros(out_shape, dtype=image_matrix.dtype)
        else:
            raise RuntimeError(f"Stretch get unsupported type: {type(stretched_matrix)}")

        # 映射
        try:
            map_coordinates(image_matrix, 
                            self.proj_map.transpose(2,0,1), 
                            output=stretched_matrix, 
                            mode='constant', 
                            cval=pad_val if type=='img' else seg_pad_val, 
                            prefilter=True)
        except Exception as e:
            pdb.set_trace()

        if isinstance(image_matrix, Tensor):
            stretched_matrix = torch.from_numpy(stretched_matrix).to(dtype=image_matrix.dtype,
                                                                     non_blocking=True)
        return stretched_matrix

    def calculate_density_factor_map(self):
        """
        Compute the information density factor map for each pixel.
        Information density factor = inverse of stretch factor along that direction.
        """
        print_log("[Dataset] Computing information density factor map", "current", logging.INFO)
        map_height, map_width = self.in_array_shape
        density_map = np.zeros(shape=(map_height, map_width), dtype=np.float32)
        # Mapping matrix pole defaults to center
        map_center_y, map_center_x = self.in_array_shape[0]/2, self.in_array_shape[1]/2
        # Traverse each pixel to compute its information density factor
        for y in range(map_height):
            for x in range(map_width):
                # Get true Y (coordinate system conversion)
                true_y = self.in_array_shape[0] - y
                # Obtain polar coordinates in mapping matrix
                radius, angle = rectangular_to_polar(x, true_y, map_center_x, map_center_y)
                # Stretch factor for this angle
                stretch_factor_value = self.stretch_factor(angle)
                # Information density factor is inverse of stretch factor
                density_factor = 1.0 / stretch_factor_value
                # Store into density map
                density_map[y, x] = density_factor
        
        print_log("[Dataset] Information density factor map completed", "current", logging.INFO)
        return density_map

    def get_density_factor_map(self):
        """
        Get (or compute if absent) the information density factor map.
        Returns a numpy array with same size as input image.
        """
        if not hasattr(self, '_density_map'):
            self._density_map = self.calculate_density_factor_map()
        return self._density_map


class LoadDensityMap(BaseTransform):
    def transform(self, results:dict):
        density_map_path = results['img_path'].replace('image', "density")
        if os.path.exists(density_map_path):
            results["density"] = cv2.imread(density_map_path, cv2.IMREAD_UNCHANGED)
            results["seg_fields"].append("density")
        return results


class PackSegInputs(_PackSegInputs):
    def transform(self, results:dict):
        packed_results = super().transform(results)
        
        if "density" in results.keys():
            density:np.ndarray = results["density"]
            if density.ndim == 2:
                density = np.expand_dims(density, -1)
            density = torch.from_numpy(density.transpose(2, 0, 1))
            packed_results['data_samples'].set_field(density, "density")
            
        return packed_results

