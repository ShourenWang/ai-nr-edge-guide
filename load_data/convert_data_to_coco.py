import json
import os
#-------------------------------------------------------#
#   用于处理COCO数据集，根据json文件生成txt文件用于训练
#-------------------------------------------------------#
import json
import os
from collections import defaultdict
import glob

#-------------------------------------------------------#
#   指向了COCO训练集与验证集图片的路径
#-------------------------------------------------------#
val_datasets_path       = "/home/shared1/nightowls_validation"
val_annotation_path     = "/home/shared1/nightowls_validation.json"

#-------------------------------------------------------#
#   生成的txt文件路径
#-------------------------------------------------------#
val_output_path         = "./coco_val.txt"
val_select_output_path  = "./coco_val_select.txt"
val_select_denoi_output_path = "./coco_val_select_denoi.txt"
selectsimg = [temp.split('/')[-1] for temp in glob.glob('/home/shared1/nightowls_validation_denoi/*.png')]
selectpath = val_datasets_path
denoipath = '/home/shared1/nightowls_validation_denoi/'
dataset = {}
# print(selects)
if __name__ == "__main__":

    name_box_id = defaultdict(list)
    id_name     = dict()
    f           = open(val_annotation_path, encoding='utf-8')
    data        = json.load(f)

    annotations = data['annotations'] #标注
    images = data['images'] #文件名
    id_filename = dict()
    for img in images:
        id = img['id']
        filename = img['file_name']
        id_filename[id] = filename

    ''' select value set'''
    for key in name_box_id.keys():
        imgname = key.split('/')[-1]

        if imgname in selectsimg:
            box_infos = name_box_id[key]
            for info in box_infos:
                x_min = int(info[0][0])
                y_min = int(info[0][1])
                x_max = x_min + int(info[0][2])
                y_max = y_min + int(info[0][3])

                box_info = " %d,%d,%d,%d,%d" % (
                    x_min, y_min, x_max, y_max, int(info[1]))


            dataset.setdefault("images",[]).append({
                'file_name': image_id,
                'id': id,
                'width': int(1024),
                'height': int(640)
                })
            dataset.setdefault("annotations",[]).append({
                'image_id': int(count),
                'bbox': [int(xmin), int(ymin), int(xmax)-int(xmin), int(ymax)-int(ymin)],
                'category_id': 6,
                        'area':int(w) * int(h),
                        'iscrowd':0,
                        'id':int(count),
                        'segmentation':[]
                })

json_name = os.path.join(folder+'instances_minival2014.json')
with open(json_name, 'w') as f:
    json.dump(dataset, f)
