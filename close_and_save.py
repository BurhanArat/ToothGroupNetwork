import pymeshfix
import os
from pymeshfix._meshfix import PyTMesh
import trimesh
import argparse

parser = argparse.ArgumentParser(description='Closing Operation with Pymeshfix')
parser.add_argument('--input_path', required=True  ,type=str,help = "input data folder path")
parser.add_argument('--output_path', required=True  ,type=str,help = "input data folder path")
parser.add_argument('--data_extension', default='.obj'  ,type=str,help = "input data folder path")
args = parser.parse_args()

dir_list = os.listdir(args.input_path)

for dirs in dir_list:

        if not os.path.exists(os.path.join(args.output_path,dirs)):
            os.makedirs(os.path.join(args.output_path,dirs))
        
        sub_dirs = os.listdir(os.path.join(args.input_path,dirs))
        for sub_dir in sub_dirs:
            if((dirs + '_lower.stl') == sub_dir):
                if not os.path.exists(os.path.join(args.output_path,dirs)+ '_lower.obj'):
                    name = sub_dir.split('.')[0]
                    mesh = trimesh.load(os.path.join(args.input_path,os.path.join(dirs,sub_dir)))

                    if mesh.is_watertight:
                        mesh.export(os.path.join(args.output_path,os.path.join(dirs,name)+args.data_extension))
                        print(name +' is watertight')
                    else:
                        tin = PyTMesh(False)

                        tin.load_file(os.path.join(args.input_path,os.path.join(dirs,sub_dir)))
                        tin.fill_small_boundaries(nbe=0, refine=True)
                        #tin.clean(max_iters=3, inner_loops=1)
                        tin.save_file(os.path.join(args.output_path,os.path.join(dirs,name)+args.data_extension))
                        print(name +' is closed')
                        del tin