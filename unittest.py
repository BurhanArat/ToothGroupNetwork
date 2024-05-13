import unittest

class EnvironmentImportTest(unittest.TestCase):
    
    def test_imports_inference(self):
        # Test import of custom module
        try:
            from inference_pipelines.inference_pipeline_maker import make_inference_pipeline
        except Exception as e:
            self.fail(f"Failed to import one or more libraries: {e}")

    def test_imports_3d(self):
        # Test import of 3D processing libraries
        try:
            import pandas as pd
            import trimesh
            import open3d 
        except Exception as e:
            self.fail(f"Failed to import one or more libraries: {e}")
        # Test import of utility libraries

    
    def test_torch(self):
        import torch
        self.assertTrue(torch.cuda.is_available(), "CUDA is not available")
        self.assertGreater(torch.cuda.device_count(), 0, "No CUDA devices found")

if __name__ == '__main__':
    unittest.main()
