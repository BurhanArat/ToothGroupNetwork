import unittest
import trimesh
import os
import open3d as o3d
import numpy as np

class EnvironmentImportTest(unittest.TestCase):
    
    def test_case(self):
    # In some cases, trimesh can change vertex order
        path = '/home/burhan_arat/test/deneme.obj'
        path_target='/home/burhan_arat/test/'
        jaw='lower'
        tri_mesh_loaded_mesh = trimesh.load_mesh(path, process=False)

        vertex_ls = np.array(tri_mesh_loaded_mesh.vertices)
        tri_ls = np.array(tri_mesh_loaded_mesh.faces)+1


        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertex_ls)
        mesh.triangles = o3d.utility.Vector3iVector(np.array(tri_ls)-1)
        mesh.compute_vertex_normals()



        id=os.path.splitext(os.path.basename(path))[0]
        target_file_path=os.path.join(path_target,id+".json")
        done=True

        pred_result=pipeline(mesh)

        pred_output = {'id_patient': id,
                        'jaw': jaw,
                        'labels': pred_result["sem"],
                        'instances': pred_result["ins"]
                        }
        with open(target_file_path, 'w') as fp:
            json.dump(pred_output, fp)


        
        
if __name__ == '__main__':
    unittest.main()