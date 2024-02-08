import sys
import os
sys.path.append(os.getcwd())
from inference_pipelines.inference_pipeline_maker import make_inference_pipeline
from glob import glob
import argparse
from predict_utils import ScanSegmentation
import math
from trimesh import transformations
import trimesh

parser = argparse.ArgumentParser(description='Inference models')
#parser.add_argument('--input_dir_path', default="G:/tooth_seg/main/all_datas/chl/3D_scans_per_patient_obj_files", type=str, help = "input directory path that contain obj files.")
parser.add_argument('--split_txt_path', default="base_name_test_fold.txt" ,type=str,help = "split txt path.")
parser.add_argument('--save_path', type=str, default="test_results", help = "result save directory.")
parser.add_argument('--model_name', type=str, default="tgnet", help = "model name. list: tsegnet | tgnet | pointnet | pointnetpp | dgcnn | pointtransformer")
parser.add_argument('--checkpoint_path', default="ckpts/tgnet_fps" ,type=str,help = "checkpoint path.")
parser.add_argument('--checkpoint_path_bdl', default="ckpts/tgnet_bdl" ,type=str,help = "checkpoint path(for tgnet_bdl).")
parser.add_argument('--save_path_pre', required=True ,type=str,help = "save path for preprocessed obj's and basename.txt")
parser.add_argument('--orig_input_lower', required=True,type=str,help = "lower STL inputs")
parser.add_argument('--orig_input_upper', required=True,type=str,help = "upper STL inputs")
args = parser.parse_args()


def preprocess(orig_input_lower,orig_input_upper,save_path_pre,split_txt_path,save_path):

    if not os.path.exists(save_path_pre):
      os.mkdir(save_path_pre)


    dir_list = os.listdir(orig_input_lower)

    angle = math.pi/18
    direction_y = [0, 1, 0]
    center = [0, 0, 0]
    
    print(len(dir_list))
    for dirs in dir_list:

        json = dirs.replace('.obj', '.json')
        if not os.path.exists(os.path.join(save_path,json)):

          name = dirs.split('_')[0]
        
          mesh_lower = trimesh.load(os.path.join(orig_input_lower,name)+'_lower.obj')
          mesh_lower.export(os.path.join(save_path_pre,name)+'_lower.obj')
          del mesh_lower
        else:
          print('skipped -- ', dirs)
        
        if not os.path.exists(os.path.join(save_path.replace('/lower/', '/upper/'),json.replace('_lower','_upper'))):
          name = dirs.split('_')[0]
          mesh_upper = trimesh.load(os.path.join(orig_input_upper,name)+'_upper.obj')

          rot_matrix_y = trimesh.transformations.rotation_matrix(angle*18, direction_y, center)

          mesh_upper.apply_transform(rot_matrix_y)

          
          mesh_upper.export(os.path.join(save_path_pre,name)+'_upper.obj')
          del mesh_upper
        else:
          print('skipped -- ', dirs.replace('_lower', '_upper'))

          
          

    text_file = open(os.path.join(split_txt_path, 'basename.txt'), "a")
    txt_dirs = os.listdir(args.save_path_pre)
    c = 0
    for dirs in txt_dirs:
      if '_lower' in dirs:
        name = dirs.split('_')[0]
        text_file.write(name)
        text_file.write('\n')
        c += 1
    print(c)
    text_file.close()


preprocess(args.orig_input_lower, args.orig_input_upper, args.save_path_pre, args.split_txt_path, args.save_path)

split_base_name_ls = []
if args.split_txt_path != "":
    f = open(os.path.join(args.split_txt_path,'basename.txt'), 'r')
    while True:
        line = f.readline()
        if not line: break
        split_base_name_ls.append(line.strip())

    f.close()


stl_path_ls = []
dir_paths = os.listdir(args.save_path_pre)
for dir_path in dir_paths:
    name = dir_path.split('_')[0]
    if name in split_base_name_ls: 
        stl_path_ls.append(os.path.join(args.save_path_pre,dir_path))

print(stl_path_ls)
pred_obj = ScanSegmentation(make_inference_pipeline(args.model_name, [args.checkpoint_path+".h5", args.checkpoint_path_bdl+".h5"]))
os.makedirs(args.save_path, exist_ok=True)
for i in range(len(stl_path_ls)):

      print(f"Processing: ", i,":",stl_path_ls[i])
      base_name = os.path.basename(stl_path_ls[i]).split(".")[0]
      print('base_name',base_name)
      if '_lower' in base_name:
        pred_obj.process(stl_path_ls[i], os.path.join(args.save_path, os.path.basename(stl_path_ls[i]).replace(".obj", ".json")))
      elif '_upper' in base_name:
        save_path = os.path.join(args.save_path, os.path.basename(stl_path_ls[i])).replace('/lower/', '/upper/')
        pred_obj.process(stl_path_ls[i], save_path.replace(".obj", ".json"))
