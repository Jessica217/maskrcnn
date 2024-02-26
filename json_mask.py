import numpy as np
import json
import cv2
import os
import glob

def get_file_pos(folder_path):
    masks_list = []
    json_files = glob.glob(os.path.join(folder_path, '*.json'))
    for json_file in json_files:
        # print(json_file)
        # json_path = './datasets/extra_DMSA_VAL/RGB_mode/0001.json'
        with open(json_file, 'r') as f:
            points = json.load(f)
            # print(points["imageHeight"], points["imageWidth"])
            left_kidney_pos = points["shapes"][0]["points"]
            right_kidney_pos = points["shapes"][1]["points"]

        img = np.zeros((points["imageHeight"], points["imageWidth"]), dtype = np.uint8)# 使用cv2创建与原图尺寸相同的图像
        pts1 = np.array(left_kidney_pos, np.int32)
        pts2 = np.array(right_kidney_pos, np.int32)
        cv2.fillPoly(img, [pts1], color = 255)
        cv2.fillPoly(img, [pts2], color = 255)
        # print(img)
        masks = np.where(img == 255, True, False)
        masks_list.append(masks)
        # print(masks_list)
        # print(masks.shape)
        # print(np.sum(masks == True))
        '''cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()'''
        yield masks


# get_file_pos(folder_path = './datasets/extra_DMSA_VAL/RGB_mode/')

