import unittest

class EnvironmentImportTest(unittest.TestCase):
    
    def test_imports(self):
        # Test import of custom module
        with self.subTest("inference_pipeline_maker"):
            from inference_pipelines.inference_pipeline_maker import make_inference_pipeline

        # Test import of common data handling and computational libraries
        with self.subTest("pandas"):
            import pandas as pd
        with self.subTest("numpy"):
            import numpy as np
        with self.subTest("json"):
            import json
        with self.subTest("math"):
            import math
        with self.subTest("os"):
            import os
        with self.subTest("argparse"):
            import argparse

        # Test import of 3D processing libraries
        with self.subTest("trimesh"):
            import trimesh
        with self.subTest("open3d"):
            import open3d as o3d

        # Test import of utility libraries
        with self.subTest("tqdm"):
            from tqdm import tqdm
        with self.subTest("torch"):
            import torch
            self.assertTrue(torch.cuda.is_available(), "CUDA is not available")
            self.assertGreater(torch.cuda.device_count(), 0, "No CUDA devices found")

if __name__ == '__main__':
    unittest.main()
