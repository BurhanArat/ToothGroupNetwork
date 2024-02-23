from inference_pipelines.inference_pipeline_maker import make_inference_pipeline
import pandas as pd
import json
import trimesh
import numpy as np
import math
import os
import open3d as o3d
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--checkpoint_path', type=str, required=True, help='checkpoint_path it should tgnet_fps.h5')
parser.add_argument('--checkpoint_path_bdl', type=str, required=True, help='checkpoint_path_bdl it should be tgnet_bdl.h5')
parser.add_argument('--path_target', type=str, required=True, help='Target Directory ')
parser.add_argument('--path_df', type=str, required=True, help='Path DataFrame')
parser.add_argument('--log_path', type=str, required=True, help="Log Path")

opt = parser.parse_args()
#!python segmentation.py  --path_df="segmentation.csv" --path_target="/content/" --checkpoint_path_bdl="/content/" --checkpoint_path="/content/ckpts(new)/tgnet_fps.h5" --checkpoint_path_bdl="/content/ckpts(new)/tgnet_bdl.h5"


# Create the pandas DataFrame
"""
DataFrame Should Be


|-- path_model   | jaw |
|                |     |


"""
model_name="tgnet"

df=pd.read_csv(opt.path_df, dtype ='str')

angle = math.pi/18
direction_y = [0, 1, 0]
center = [0, 0, 0]

pipeline=make_inference_pipeline(model_name, [opt.checkpoint_path, opt.checkpoint_path_bdl])
def load_obj(path,jaw):
    # In some cases, trimesh can change vertex order

    tri_mesh_loaded_mesh = trimesh.load_mesh(path, process=False)
    if jaw=="upper":
      rot_matrix_y = trimesh.transformations.rotation_matrix(angle*18, direction_y, center)
      tri_mesh_loaded_mesh.apply_transform(rot_matrix_y)
    vertex_ls = np.array(tri_mesh_loaded_mesh.vertices)
    tri_ls = np.array(tri_mesh_loaded_mesh.faces)+1


    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertex_ls)
    mesh.triangles = o3d.utility.Vector3iVector(np.array(tri_ls)-1)
    mesh.compute_vertex_normals()

    return mesh


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

for row in tqdm(df.iterrows(),desc="Segmentation"):
    path_model=str(row[1]["path_model"])
    jaw=str(row[1]["jaw"])
    id=os.path.splitext(os.path.basename(path_model))[0]
    target_file_path=os.path.join(opt.path_target,id+".json")
    done=True
    try:

        mesh=load_obj(path_model,jaw=jaw)

        pred_result=pipeline(mesh)

        if jaw=="upper":
            pred_result["sem"][pred_result["sem"]>0] += 20

        pred_output = {'id_patient': id,
                        'jaw': jaw,
                        'labels': pred_result["sem"],
                        'instances': pred_result["ins"]
                        }
        with open(target_file_path, 'w') as fp:
        json.dump(pred_output, fp, cls=NpEncoder)
    except Exception as e :
        done=False
        print(e)

    log={
        
         "done" : done,
         "id" : data["id"]
    }
    logs.append(log)
    df_logs=pd.DataFrame(logs)

    df_logs.to_csv(os.path.join(opt.log_path,"./logs.csv"),index=False)


