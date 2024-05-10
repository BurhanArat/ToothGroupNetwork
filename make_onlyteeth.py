import os
import json
import trimesh
import numpy as np

mesh_path = "C:/Users/pc/Desktop/data_from_niels"
mesh_label = 'C:/Users/pc/Desktop/segm_results/label'
result_path = 'C:/Users/pc/Desktop/segm_results/models'

dir_lists = os.listdir(mesh_path)

for dirs in dir_lists:
    sub_dirs = os.listdir(os.path.join(mesh_path,dirs))
    for sub_dir in sub_dirs:
        if (dirs + '_UpperJaw.stl') == sub_dir:
            
            mesh = trimesh.load(os.path.join(mesh_path,os.path.join(dirs,sub_dir)))
            f = open(mesh_label + '/' + dirs + '_UpperJaw.json')
            json_data = json.load(f)
            labels = np.float32(np.asarray(json_data['labels']))
    
    
            #Removing teeths from original models
            mask=np.ones_like(labels)
            mask=labels>1
            asd = mesh.faces
            face_onlyteeth_mask = mask[mesh.faces].all(axis=1)
            mesh.update_faces(face_onlyteeth_mask)
            mesh.export(os.path.join(result_path,dirs)+'_upper_onlyteeth.obj')
            
        elif(dirs + '_LowerJaw.stl') == sub_dir:
                mesh = trimesh.load(os.path.join(mesh_path,os.path.join(dirs,sub_dir)))
                f = open(mesh_label + '/' + dirs + '_LowerJaw.json')
                json_data = json.load(f)
                labels = np.float32(np.asarray(json_data['labels']))
        
        
                mask=np.ones_like(labels)
                mask=labels>1
                asd = mesh.faces
                face_onlyteeth_mask = mask[mesh.faces].all(axis=1)
                mesh.update_faces(face_onlyteeth_mask)
                mesh.export(os.path.join(result_path,dirs)+'_lower_onlyteeth.obj')
