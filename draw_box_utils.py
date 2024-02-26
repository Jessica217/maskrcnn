import numpy
from PIL.Image import Image, fromarray
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from PIL import ImageColor
import numpy as np
import cv2
from json_mask import get_file_pos

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def draw_text(draw,
              box: list,
              cls: int,
              score: float,
              category_index: dict,
              color: str,
              font: str = 'arial.ttf',
              font_size: int = 18):
    """
    将目标边界框和类别信息绘制到图片上
    """
    try:
        font = ImageFont.truetype(font, font_size)
    except IOError:
        font = ImageFont.load_default()

    left, top, right, bottom = box
    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str = f"{category_index[str(cls)]}: {int(100 * score)}%"
    display_str_heights = [font.getsize(ds)[1] for ds in display_str]
    # Each display_str has a top and bottom margin of 0.05x.
    display_str_height = (1 + 2 * 0.05) * max(display_str_heights)

    if top > display_str_height:
        text_top = top - display_str_height
        text_bottom = top
    else:
        text_top = bottom
        text_bottom = bottom + display_str_height

    for ds in display_str:
        text_width, text_height = font.getsize(ds)
        margin = np.ceil(0.05 * text_width)
        draw.rectangle([(left, text_top),
                        (left + text_width + 2 * margin, text_bottom)], fill=color)
        draw.text((left + margin, text_top),
                  ds,
                  fill='black',
                  font=font)
        left += text_width


def draw_masks(image, masks, colors, list, mask_GT, thresh: float = 0.7, alpha: float = 0.5):

    np_image = np.array(image) # 原图尺寸
    masks = np.where(masks > thresh, True, False) # 肾脏
    masks_left = masks[0]
    masks_right = masks[1]
    mask_DL = masks_left + masks_right
    '''print(mask_all)
    print(mask_all.shape)'''
    #mask_DL = np.sum(mask_all == True)
    # print(mask_DL.shape)
    # print(True_num_dl)
    # get_file_pos('./datasets/extra_DMSA_VAL/RGB_mode/') # 调用函数
    # print(mask_GT.shape)
    intersect = masks * mask_DL
    union = mask_GT + mask_DL
    inter_num = np.sum(intersect == True)
    union_num = np.sum(union == True)
    iou = inter_num / union_num
    list.append(iou)

    # colors = np.array(colors)
    img_to_draw = np.copy(np_image) # 创建图像副本
    # TODO: There might be a way to vectorize this
    for mask, color in zip(masks, colors):
        img_to_draw[mask] = color

    out = np_image * (1 - alpha) + img_to_draw * alpha
    return fromarray(out.astype(np.uint8))


from PIL import Image, ImageDraw, ImageFont
import numpy as np
from typing import List

def draw_objs(image: Image,
              mask,
              list_iou,
              boxes: np.ndarray = None,
              classes: np.ndarray = None,
              scores: np.ndarray = None,
              masks: np.ndarray = None,
              category_index: dict = None,
              box_thresh: float = 0.1,
              mask_thresh: float = 0.5,
              line_thickness: int = 8,
              font: str = 'arial.ttf',
              font_size: int = 18,
              draw_boxes_on_image: bool = True,
              draw_masks_on_image: bool = True,
              nms_thresh: float = 0.5):
    """
    将目标边界框信息，类别信息，mask信息绘制在图片上
    Args:
        image: 需要绘制的图片
        boxes: 目标边界框信息
        classes: 目标类别信息
        scores: 目标概率信息
        masks: 目标mask信息
        category_index: 类别与名称字典
        box_thresh: 过滤的概率阈值
        mask_thresh:
        line_thickness: 边界框宽度
        font: 字体类型
        font_size: 字体大小
        draw_boxes_on_image:
        draw_masks_on_image:
        nms_thresh: 非极大值抑制的阈值

    Returns:

    """
    # 过滤掉低概率的目标
    idxs = np.greater(scores, box_thresh)
    boxes = boxes[idxs]
    classes = classes[idxs]
    scores = scores[idxs]
    if masks is not None:
        masks = masks[idxs]
    if len(boxes) == 0:
        return image

    # 使用非极大值抑制过滤重叠的框
    selected_indices = non_max_suppression(boxes, scores, nms_thresh)
    boxes = boxes[selected_indices]
    classes = classes[selected_indices]
    scores = scores[selected_indices]
    if masks is not None:
        masks = masks[selected_indices]

    #colors = [ImageColor.getrgb(STANDARD_COLORS[cls % len(STANDARD_COLORS)]) for cls in classes]
    colors = [ImageColor.getrgb('CornflowerBlue'), ImageColor.getrgb('Cyan')]

    if draw_boxes_on_image:
        # Draw all boxes onto image.
        draw = ImageDraw.Draw(image)
        for box, cls, score, color in zip(boxes, classes, scores, colors):
            left, top, right, bottom = box
            # 绘制目标边界框
            draw.line([(left, top), (left, bottom), (right, bottom),
                       (right, top), (left, top)], width=line_thickness, fill=color)
            # 绘制类别和概率信息
            draw_text(draw, box.tolist(), int(cls), float(score), category_index, color, font, font_size)

    if draw_masks_on_image and (masks is not None):
        # Draw all mask onto image.
        image = draw_masks(image, masks, colors, list_iou, mask,  mask_thresh)

    return image


# 使用非极大值抑制
def non_max_suppression(boxes, scores, threshold):
    """
    非极大值抑制
    Args:
        boxes: 目标边界框
        scores: 目标概率
        threshold: IoU 阈值

    Returns:
        保留的边界框索引

    """
    if len(boxes) == 0:
        return []

    # 计算所有框的面积
    areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)

    # 根据分数降序排序框的索引
    order = np.argsort(scores)[::-1]

    # 保留的索引
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        # 计算当前框与其他框的IoU
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)

        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= threshold)[0]
        order = order[inds + 1]

    return keep


def bbox_iou(box1, box2):
    """
    计算两个边界框的 IoU
    Args:
        box1: 边界框1
        box2: 边界框2

    Returns:
        IoU 值

    """
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    xi = max(x1, x3)
    yi = max(y1, y3)
    wi = max(0, min(x2, x4) - xi)
    hi = max(0, min(y2, y4) - yi)

    inter_area = wi * hi
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)

    iou = inter_area / float(box1_area + box2_area - inter_area + 1e-16)

    return iou

# 其他辅助函数的定义和引用（draw_text, draw_masks等）没有提供，你需要在代码中实现或者引入相应的函数。

