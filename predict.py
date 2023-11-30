import os
import time
import json

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

from network_files import MaskRCNN
from backbone import resnet50_fpn_backbone
from draw_box_utils import draw_objs


def create_model(num_classes, box_thresh=0.7):
    backbone = resnet50_fpn_backbone()
    model = MaskRCNN(backbone,
                     num_classes=num_classes,
                     rpn_score_thresh=box_thresh,
                     box_score_thresh=box_thresh)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def batch_inference(input_folder, output_folder, num_classes=2, box_thresh=0.5, weights_path="./save_weights/best_model_for_64_2(augdata).pth", label_json_path='coco91_indices.json'):
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device.".format(device))

    # create model
    model = create_model(num_classes=num_classes + 1, box_thresh=box_thresh)

    # load train weights
    assert os.path.exists(weights_path), "{} file does not exist.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)

    # read class_indict
    assert os.path.exists(label_json_path), "JSON file {} does not exist.".format(label_json_path)
    with open(label_json_path, 'r') as json_file:
        category_index = json.load(json_file)

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            # load image
            img_path = os.path.join(input_folder, filename)
            original_img = Image.open(img_path).convert('RGB')

            # from PIL image to tensor, do not normalize image
            data_transform = transforms.Compose([transforms.ToTensor()])
            img = data_transform(original_img)
            # expand batch dimension
            img = torch.unsqueeze(img, dim=0)

            model.eval()  # 进入验证模式
            with torch.no_grad():
                # init
                img_height, img_width = img.shape[-2:]
                init_img = torch.zeros((1, 3, img_height, img_width), device=device)
                model(init_img)

                t_start = time_synchronized()
                predictions = model(img.to(device))[0]
                t_end = time_synchronized()
                print("Inference+NMS time: {}".format(t_end - t_start))

                predict_boxes = predictions["boxes"].to("cpu").numpy()
                predict_classes = predictions["labels"].to("cpu").numpy()
                predict_scores = predictions["scores"].to("cpu").numpy()
                predict_mask = predictions["masks"].to("cpu").numpy()
                predict_mask = np.squeeze(predict_mask, axis=1)  # [batch, 1, h, w] -> [batch, h, w]

                if len(predict_boxes) == 0:
                    print("No objects detected in {}!".format(filename))
                    continue

                plot_img = draw_objs(original_img,
                                     boxes=predict_boxes,
                                     classes=predict_classes,
                                     scores=predict_scores,
                                     masks=predict_mask,
                                     category_index=category_index,
                                     line_thickness=3,
                                     font='arial.ttf',
                                     font_size=20)
                # plt.imshow(plot_img)
                # plt.show()

        # Save predicted image result_output
        output_path = os.path.join(output_folder, f"result_{filename}")
        plot_img.save(output_path)
        print(f"Saved: {output_path}")


if __name__ == '__main__':
    input_folder = './datasets/coco_aug/val2017'  # Replace with the actual input folder path
    output_folder = './datasets/result_output_for_64_2_aug_nms'  # Replace with the actual output folder path

    batch_inference(input_folder, output_folder)
