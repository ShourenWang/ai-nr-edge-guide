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
train_datasets_path     = "/home/shared1/nightowls_training"
val_datasets_path       = "/home/shared1/nightowls_validation"

#-------------------------------------------------------#
#   指向了COCO训练集与验证集标签的路径
#-------------------------------------------------------#
train_annotation_path   = "/home/shared1/nightowls_training.json"
val_annotation_path     = "/home/shared1/nightowls_validation.json"

#-------------------------------------------------------#
#   生成的txt文件路径
#-------------------------------------------------------#
train_output_path       = "./coco_train.txt"
val_output_path         = "./coco_val.txt"
val_select_output_path         = "./coco_val_select.txt"
val_select_denoi_output_path         = "./coco_val_select_denoi.txt"
selectsimg = [temp.split('/')[-1] for temp in glob.glob('/home/shared1/nightowls_validation_denoi/*.png')]
selectpath = val_datasets_path
denoipath = '/home/shared1/nightowls_validation_denoi/'
# print(selects)
if __name__ == "__main__":
    name_box_id = defaultdict(list)
    id_name     = dict()
    f           = open(train_annotation_path, encoding='utf-8')
    data        = json.load(f)

    annotations = data['annotations']
    images = data['images']  # 文件名
    '''
    images      的 id表示图片
    annotations 的 images_id 表示一张图，id不知道干嘛，不用的
    '''
    id_filename = dict()
    for img in images:
        id = img['id']
        filename = img['file_name']
        id_filename[id] = filename

    for ant in annotations:
        id = ant['image_id']
        name = os.path.join(train_datasets_path, id_filename[id])
        cat = ant['category_id']
        if cat >= 1 and cat <= 11:
            cat = cat - 1
        elif cat >= 13 and cat <= 25:
            cat = cat - 2
        elif cat >= 27 and cat <= 28:
            cat = cat - 3
        elif cat >= 31 and cat <= 44:
            cat = cat - 5
        elif cat >= 46 and cat <= 65:
            cat = cat - 6
        elif cat == 67:
            cat = cat - 7
        elif cat == 70:
            cat = cat - 9
        elif cat >= 72 and cat <= 82:
            cat = cat - 10
        elif cat >= 84 and cat <= 90:
            cat = cat - 11

        name_box_id[name].append([ant['bbox'], cat])

    f = open(train_output_path, 'w')
    for key in name_box_id.keys():
        f.write(key)
        box_infos = name_box_id[key]
        for info in box_infos:
            x_min = int(info[0][0])
            y_min = int(info[0][1])
            x_max = x_min + int(info[0][2])
            y_max = y_min + int(info[0][3])

            box_info = " %d,%d,%d,%d,%d" % (
                x_min, y_min, x_max, y_max, int(info[1]))

            f.write(box_info)
        f.write('\n')
    f.close()

    '''
    make value set
    '''

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
    cats = []
    for ant in annotations:
        id = ant['image_id']
        name = os.path.join(val_datasets_path, id_filename[id] )
        cat = ant['category_id']
        cats.append(cat)
        if cat >= 1 and cat <= 11:
            cat = cat - 1
        elif cat >= 13 and cat <= 25:
            cat = cat - 2
        elif cat >= 27 and cat <= 28:
            cat = cat - 3
        elif cat >= 31 and cat <= 44:
            cat = cat - 5
        elif cat >= 46 and cat <= 65:
            cat = cat - 6
        elif cat == 67:
            cat = cat - 7
        elif cat == 70:
            cat = cat - 9
        elif cat >= 72 and cat <= 82:
            cat = cat - 10
        elif cat >= 84 and cat <= 90:
            cat = cat - 11
        name_box_id[name].append([ant['bbox'], cat])
    print(set(cats))

    f = open(val_output_path, 'w')

    for key in name_box_id.keys():
        f.write(key)
        box_infos = name_box_id[key]
        for info in box_infos:
            x_min = int(info[0][0])
            y_min = int(info[0][1])
            x_max = x_min + int(info[0][2])
            y_max = y_min + int(info[0][3])

            box_info = " %d,%d,%d,%d,%d" % (
                x_min, y_min, x_max, y_max, int(info[1]))
            f.write(box_info)
        f.write('\n')
    f.close()

    # ''' select value set'''
    #
    # f_select = open(val_select_output_path, 'w')
    # f_denoi = open(val_select_denoi_output_path, 'w')
    # print(len(name_box_id))
    # for key in name_box_id.keys():
    #     imgname = key.split('/')[-1]
    #
    #     if imgname in selectsimg:
    #         print(imgname)
    #
    #         f_denoi.write(denoipath+imgname)
    #         f_select.write(selectpath + '/' + imgname)
    #         box_infos = name_box_id[key]
    #         for info in box_infos:
    #             x_min = int(info[0][0])
    #             y_min = int(info[0][1])
    #             x_max = x_min + int(info[0][2])
    #             y_max = y_min + int(info[0][3])
    #
    #             box_info = " %d,%d,%d,%d,%d" % (
    #                 x_min, y_min, x_max, y_max, int(info[1]))
    #             f_denoi.write(box_info)
    #             f_select.write(box_info)
    #         f_denoi.write('\n')
    #         f_select.write('\n')
    # f_denoi.close()
    # f_select.close()
