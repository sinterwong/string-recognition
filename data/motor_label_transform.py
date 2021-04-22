from shapely.geometry import Polygon,MultiPoint,LineString
import sys
import json
import os
import glob
import sys
import shutil
from shapely.geometry import Polygon
import shapely
from shapely.geometry import Polygon, mapping
import cv2
import operator
from functools import reduce
import numpy as np
import random

'''
转换检测格式 仅保留骑行检测
'''

def inclusion_ratio(bbox1, bbox2):
    '''
    计算bbox2 被 bbox1 包含的程度 (重合的面积 / bbox2 的面积)
    :param bbox1: [x1, y1, x2, y2]
    :param bbox2: [x1, y1, x2, y2] or poly
    
    return
    '''
    assert (type(bbox1) == list) and len(bbox1) == 4, "bbox1 must be list and length is 4"

    if type(bbox2) != list:
        bbox2 = bbox2.astype(np.int32)
        bbox2 = [np.min(bbox2[:, 0]), np.min(bbox2[:, 1]), np.max(bbox2[:, 0]), np.max(bbox2[:, 1])]

    # bbox2 的面积
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    ix1 = max(bbox1[0], bbox2[0])
    iy1 = max(bbox1[1], bbox2[1])
    ix2 = min(bbox1[2], bbox2[2])
    iy2 = min(bbox1[3], bbox2[3])
    # 重合部分的面积
    intersectio_area = max(0, (ix2 - ix1)) * max(0, (iy2 - iy1))

    ratio = intersectio_area / bbox2_area

    return ratio


def format_bbox(points):
    x1 = min(points[0][0], points[1][0])
    y1 = min(points[0][1], points[1][1])
    x2 = max(points[0][0], points[1][0])
    y2 = max(points[0][1], points[1][1])

    return list(map(lambda x: round(x), [x1, y1, x2, y2]))


def format_poly(points):
    '''
    将 points 按照 右下，左下，左上，右上 排序
    '''
    assert len(points) == 4, "polygon doesn't fit the bill"
    x1 = points[0][0]
    x2 = points[1][0]
    x3 = points[2][0]
    x4 = points[3][0]
    y1 = points[0][1]
    y2 = points[1][1]
    y3 = points[2][1]
    y4 = points[3][1]

    poly_coords = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    sort_x_ids = np.argsort(poly_coords[:, 0])
    left_coord = poly_coords[sort_x_ids[:2]]
    right_coord = poly_coords[sort_x_ids[2:]]

    left_coord = left_coord[np.argsort(left_coord[:, 1])]  # get left up, left down
    right_coord = right_coord[np.argsort(right_coord[:, 1])]  # get right up, right down
    # 右下，左下，左上，右上
    poly_coords = np.array([right_coord[1], left_coord[1], left_coord[0], right_coord[0]]).astype(np.float32)

    return poly_coords


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def calculate_iobox2(bbox, box_2):
    try:
        
        x1 = bbox[0][0]
        y1 = bbox[0][1]

        x2 = bbox[1][0]
        y2 = bbox[1][1]
        
        poly_1 = Polygon([[x1,y1],[x1,y2],[x2,y2],[x2,y1]])
        poly_2 = Polygon(box_2)
        
        # print(poly_2.exterior.coords.xy)
        
        roi = poly_1.intersection(poly_2)
        rate = roi.area / poly_2.area
        point = []
        if type(roi) is not  shapely.geometry.collection.GeometryCollection:
            point =list(set(list(mapping(roi)['coordinates'][0])))
        
        if point == []:
            point = 0
        
        return rate, point

    except BaseException:
        
        return -1,-1


def get_riding_target(infos, size):
    
    target = ''
    if infos[0] == '非机动车':
        infos[0] = '非机动'

    if infos[0] == '机动车':
        infos[0] = '摩托车'

    label, points = infos

    x1 = min(points[0][0], points[1][0])
    y1 = min(points[0][1], points[1][1])
    x2 = max(points[0][0], points[1][0])
    y2 = max(points[0][1], points[1][1])

    if int(x2 - x1) <= 10 or int(y2 - y1) <= 10:
        return None
    
    w, h = size
    GT = convert((int(w), int(h)), [x1, x2, y1, y2])

    target += str(ridingcls2ids[label]) + '\t'

    for s in GT:
        target += str(s) + '\t'

    # print(target.strip('\t'))
    return target.strip('\t')


def get_has_target(infos, size):
    
    target = ''

    if infos[0] == '1头肩区域':
        infos[0] = '1-头肩区域'

    if infos[0] == '0头肩区域':
        infos[0] = '0-头肩区域'

    label, points = infos

    x1 = min(points[0][0], points[1][0])
    y1 = min(points[0][1], points[1][1])
    x2 = max(points[0][0], points[1][0])
    y2 = max(points[0][1], points[1][1])

    if int(x2 - x1) <= 10 or int(y2 - y1) <= 10:
        return None
    
    w, h = size
    GT = convert((int(w), int(h)), [x1, x2, y1, y2])

    target += str(ridingcls2ids[label]) + '\t'

    for s in GT:
        target += str(s) + '\t'

    # print(target.strip('\t'))
    return target.strip('\t')


def labelme_processing_riding(json_path, classes, match_lp):
    '''
    处理单张骑行标注上所有可用数据，包括：
    1、骑行检测转换成 yolo 的输出
    2、头肩转换成 yolo 的输出
    3、车牌检测训练格式
    4、车牌识别训练格式
    5、车牌模糊清晰训练格式
    6、头盔识别训练格式

    return: 所有可用信息
    '''

    # 头肩区域、车牌区域的信息需要建立在骑行的区域之上

    

    with open(json_path, 'r', encoding='gbk') as json_file:
        img = cv2.imread(json_path.replace('json', 'jpg'))
        mask = np.ones((img.shape[0], img.shape[1]))
        data = json.load(json_file)
        target_ls = []

        size = (data['imageWidth'], data['imageHeight'])

        riding_temp = filter(lambda x: x['label'] in ['非机动', '非机动车', '摩托车', '机动车'], data['shapes'])
        riding_shapes = list(map(lambda x: [x['label'], x['points']], riding_temp))

        helmet_temp = filter(lambda x: x['label'] in ['1头肩区域', '1-头肩区域', '0头肩区域', '0-头肩区域'], data['shapes'])
        helmet_shapes = list(map(lambda x: [x['label'], x['points']], helmet_temp))

        mask_temp = filter(lambda x: x['label'] == '0', data['shapes'])
        mask_shapes = list(map(lambda x: x['points'], mask_temp))

        lp_temp = filter(lambda x: (x['label'] in ['模糊', '车牌模糊']) or (x['label'][0] in match_lp), data['shapes'])
        lp_shapes = list(map(lambda x: [x['label'], x['points']], lp_temp))

        # TODO 处理掉图片上 mask 部分
        for i, mask_shape in enumerate(mask_shapes):
            cv2.fillPoly(mask, np.array(mask_shape, dtype=np.int32)[np.newaxis, :, :], 0)

            for s in range(3):
                img[:, :, s][mask == 0] = mask[mask == 0]

        riding_targets = []  # 存放每一行 yolo 形式的 target
        for i, riding_shape in enumerate(riding_shapes):
            # --------------------- 获取骑行的 target -------------------------
            riding_target = ''
            if riding_shape[0] == '非机动车':
                riding_shape[0] = '非机动'

            if riding_shape[0] == '机动车':
                riding_shape[0] = '摩托车'

            ride_label, ride_points = riding_shape

            rx1 = max(0, min(ride_points[0][0], ride_points[1][0]))
            ry1 = max(0, min(ride_points[0][1], ride_points[1][1]))
            rx2 = min(max(ride_points[0][0], ride_points[1][0]), size[0])
            ry2 = min(max(ride_points[0][1], ride_points[1][1]), size[1])
            
            if int(rx2 - rx1) <= 10 or int(ry2 - ry1) <= 10 or rx1 >= rx2 or ry1 >= ry2:
                print('错误的size')
                continue
            
            GT = convert((int(size[0]), int(size[1])), [rx1, rx2, ry1, ry2])

            riding_target += str(ridingcls2ids[ride_label]) + '\t'

            for s in GT:
                riding_target += str(s) + '\t'
            riding_targets.append(riding_target.strip('\t'))

            ride_img = img[int(ry1): int(ry2), int(rx1): int(rx2)]

            # ----------------------------获取头肩的 target ------------------------------
            HAS_targets = []
            for j, helmet_shape in enumerate(helmet_shapes):
                '''
                遍历所有的 riding 区域和 HAS 区域计算包含率, 大于 0.5 的算到当前图片中
                一边保存图片，一边保存label
                '''
                helmet_target = ''

                if helmet_shape[0] == '1头肩区域':
                    helmet_shape[0] = '1-头肩区域'

                if helmet_shape[0] == '0头肩区域':
                    helmet_shape[0] = '0-头肩区域'
                if len(helmet_shape[1]) != 4:
                    print('头肩不是四点')
                    continue

                ratio, helmet_points = calculate_iobox2(ride_points, helmet_shape[1])
                if ratio == -1:
                    print('头肩区域计算有误')
                    continue
                # print(helmet_points)
                if ratio > 0.5 and len(helmet_points) == 4:
                    # 坐标偏移
                    o_point = np.array(helmet_points)

                    o_point[:, 0] -= rx1
                    o_point[:, 1] -= ry1

                    hx1 = max(0, np.min(o_point[:, 0]))
                    hx2 = min(rx2 - rx1, np.max(o_point[:, 0]))
                    hy1 = max(0, np.min(o_point[:, 1]))
                    hy2 = min(ry2 - ry1, np.max(o_point[:, 1]))

                    helmet_label = helmet_shape[0][0]

                    if not helmet_label in ['0', '1']:
                        print('error: ', helmet_shape[0])
                        continue
                    # ----------------------- 获取头盔识别的数据 ---------------------------
                    helmet_img_name = helmet_label + "_" + os.path.basename(json_path).strip('.json') + "_" + str(i) + "_" + str(j) + '.jpg' 

                    helmet_img = ride_img[int(hy1): int(hy2), int(hx1): int(hx2)]

                    if not os.path.exists(os.path.join(helmet_recognition_outdir, helmet_label)):
                        os.makedirs(os.path.join(helmet_recognition_outdir, helmet_label))
                    cv2.imwrite(os.path.join(helmet_recognition_outdir, helmet_label, helmet_img_name), helmet_img.astype(np.uint8))

                    # ----------------------- 获取头盔识别的数据 ---------------------------
                    try:
                        h_GT = convert((ride_img.shape[1], ride_img.shape[0]), [hx1, hx2, hy1, hy2])
                    except Exception as e:
                        print(e)
                        print(ride_img.shape)
                        exit()


                    helmet_target += helmet_label + '\t'

                    for s in h_GT:
                        helmet_target += str(s) + '\t'

                    HAS_targets.append(helmet_target.strip('\t'))

            pic_has_name = str(i) + '_' + os.path.basename(json_path).replace('json', 'jpg')

            if len(HAS_targets) == 1:
                cv2.imwrite(os.path.join(has_detect_outdir, pic_has_name), ride_img.astype(np.uint8))
                with open(os.path.join(has_detect_outdir, pic_has_name.replace('jpg', 'txt')), 'w', encoding='utf8') as wf:
                    for t in HAS_targets:
                        wf.write(t)
                        wf.write("\n")
            # ----------------------------获取头肩的 target ------------------------------









            # ----------------------------获取车牌的 target ------------------------------
            licence_plate_targets = []  # 存放车牌偏移后位置信息、车牌号以及是否模糊信息
            licence_plate_targets_bad = []  # 存放车牌偏移后位置信息、车牌号以及是否模糊信息
            for j, shape in enumerate(lp_shapes):
                
                lp_type = '1'
                lp_label = shape[0] 
                if lp_label == '模糊':
                    lp_label = '车牌模糊'

                if len(shape[1]) == 2:
                    tx1 = shape[1][0][0]
                    ty1 = shape[1][0][1]

                    tx2 = shape[1][1][0]
                    ty2 = shape[1][1][1]
                    shape[1] = [[tx2, ty2], [tx1, ty2], [tx1, ty1], [tx2, ty1]]
                
                ratio, lp_points = calculate_iobox2(ride_points, shape[1])

                if ratio == -1:
                    print('车牌区域计算有误')
                    # cv2.imwrite('/home/wangxt/workspace/datasets/Motor/preprocessed/%05d.jpg' % random.randint(0, 999999999), ride_img)
                    continue

                if ratio > 0.3 and len(lp_points) == 4:
                    # 坐标偏移
                    o_point = np.array(lp_points)
                    o_point[:, 0] -= rx1
                    o_point[:, 1] -= ry1

                    o_point[:, 0] = np.maximum(o_point[:, 0], 0)
                    o_point[:, 0] = np.minimum(o_point[:, 0], rx2 - rx1)
                    o_point[:, 1] = np.maximum(o_point[:, 1], 0)
                    o_point[:, 1] = np.minimum(o_point[:, 1], ry2 - ry1)

                    # 排序坐标位置
                    sort_x_ids = np.argsort(o_point[:, 0])
                    left_coord = o_point[sort_x_ids[:2]]
                    right_coord = o_point[sort_x_ids[2:]]
                    left_coord = left_coord[np.argsort(left_coord[:, 1])]
                    right_coord = right_coord[np.argsort(right_coord[:, 1])]
                    # 右下，左下，左上，右上
                    o_point = np.array([right_coord[1], left_coord[1], left_coord[0], right_coord[0]]).astype(np.float32)

                    # 截取车牌
                    lx1 = max(0, np.min(o_point[:, 0]))
                    lx2 = min(rx2 - rx1, np.max(o_point[:, 0]))
                    ly1 = max(0, np.min(o_point[:, 1]))
                    ly2 = min(ry2 - ry1, np.max(o_point[:, 1]))

                    # ----------------------- 获取车牌识别的数据 ---------------------------
                    if lp_label == '车牌模糊':
                        lp_img_name = os.path.basename(json_path).strip('.json') + "_" + str(i) + "_" + str(j) + '.jpg' 
                        lp_type = '0'
                        licence_plate_targets_bad.append((lp_label, o_point))

                    else:
                        lp_img_name = lp_label + "_" + "yellow" + "_" + str(i) + "_" + str(j) + '.jpg' 
                        licence_plate_targets.append((lp_label, o_point))

                    if (len(lp_label) != 7) and lp_label != '车牌模糊':
                        lp_type = '2'
                    
                    if ratio > 0.95:
                        lp_img = ride_img[int(ly1): int(ly2), int(lx1): int(lx2)]

                        if not os.path.exists(os.path.join(lp_recognition_outdir, lp_type)):
                            os.makedirs(os.path.join(lp_recognition_outdir, lp_type))
                        cv2.imwrite(os.path.join(lp_recognition_outdir, lp_type, lp_img_name), lp_img.astype(np.uint8))

                    # ----------------------- 获取车牌识别的数据 ---------------------------

            if len(licence_plate_targets) == 1:
                pic_lp_name = str(i) + '_' + os.path.basename(json_path).replace('json', 'jpg')

                str2write = ""
                str2write += pic_lp_name
                str2write += '\t' + str(len(licence_plate_targets)) + '\t'

                ps = []
                for v, p in enumerate(list(zip(*licence_plate_targets))[1]):
                    ps += list(p.reshape(-1))
                for v, p in enumerate(ps):
                    if v == 0:
                        str2write += str(p)
                        continue
                    str2write += "," + str(p)
                
                for v, p in enumerate(list(zip(*licence_plate_targets))[0]):
                    str2write += '\t'
                    str2write += p
                print(str2write)
                cv2.imwrite(os.path.join(lp_detect_outdir, pic_lp_name), ride_img.astype(np.uint8))

                with open(os.path.join(lp_detect_outdir, pic_lp_name.replace('jpg', 'txt')), 'w', encoding='utf8') as wf:
                    wf.write(str2write)

            if len(licence_plate_targets_bad) == 1:
                pic_lp_name = str(i) + '_' + os.path.basename(json_path).replace('json', 'jpg')

                str2write = ""
                str2write += pic_lp_name
                str2write += '\t' + str(len(licence_plate_targets_bad)) + '\t'

                ps = []
                for v, p in enumerate(list(zip(*licence_plate_targets_bad))[1]):
                    ps += list(p.reshape(-1))
                for v, p in enumerate(ps):
                    if v == 0:
                        str2write += str(p)
                        continue
                    str2write += "," + str(p)

                for v, p in enumerate(list(zip(*licence_plate_targets_bad))[0]):
                    str2write += '\t'
                    str2write += p
                print(str2write)
                cv2.imwrite(os.path.join(lp_detect_bad_outdir, pic_lp_name), ride_img.astype(np.uint8))

                with open(os.path.join(lp_detect_bad_outdir, pic_lp_name.replace('jpg', 'txt')), 'w', encoding='utf8') as wf:
                    wf.write(str2write)

            # ----------------------------获取车牌的 target ------------------------------










        # 保存骑行图像
        with open(os.path.join(riding_outdir, os.path.basename(json_path.replace('.json', '.txt'))), 'w', encoding='utf8') as wf:
            for t in riding_targets:
                wf.write(t)
                wf.write("\n")
        cv2.imwrite(os.path.join(riding_outdir, os.path.basename(json_path.replace('.json', '.jpg'))), img)
        # shutil.copy(json_path.replace('.json', '.jpg'), os.path.join(riding_outdir, os.path.basename(json_path.replace('.json', '.jpg'))))

        # ----------------------------获取骑行的 target ------------------------------



def test():

    classes = ['摩托车', '非机动', '车牌模糊', '1-头肩区域', '0-头肩区域', 'mask']
    licence_plate_str = '皖沪津渝冀晋蒙辽吉黑苏浙京闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新'

    label_list = glob.glob(data_path + '/*.json')
    for i, path in enumerate(label_list):
        labelme_processing_riding(path, classes, licence_plate_str)


if __name__ == "__main__":
    '''
    data_path = "/home/wangxt/workspace/datasets/Motor/20-01-02/4"
    # classes = ['motorbike', 'electrombile']
    classes = ['摩托车', '非机动', '车牌模糊', '1-头肩区域', '0-头肩区域', 'mask']
    ids2cls = dict(enumerate(classes))
    cls2ids = {c: idx for idx, c in ids2cls.items()}
    licence_plate_str = '皖沪津渝冀晋蒙辽吉黑苏浙京闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新'
    get_yolov3_dataset(data_path, cls2ids)
    '''
    # data_path = "/home/wangxt/workspace/datasets/Motor/20-01-02/4"
    # data_path = "/data/sinter/CongHua/Done/Riding/part-1"
    data_path = "/home/wangjq/sinter/datasets/Motor-datasets"

    riding_outdir = '/home/wangjq/sinter/datasets/Motor-processed/Ride-Detect'
    has_detect_outdir = '/home/wangjq/sinter/datasets/Motor-processed/HAS-Detect'
    lp_detect_outdir = '/home/wangjq/sinter/datasets/Motor-processed/Licence-Plate-Detect'
    lp_recognition_outdir = '/home/wangjq/sinter/datasets/Motor-processed/Licence-Plate-Recognition'
    helmet_recognition_outdir = '/home/wangjq/sinter/datasets/Motor-processed/Helmet-Recognition'
    lp_detect_bad_outdir = '/home/wangjq/sinter/datasets/Motor-processed/Licence-Plate-Detect-Bad'

    if not os.path.exists(riding_outdir):
        os.makedirs(riding_outdir)

    if not os.path.exists(has_detect_outdir):
        os.makedirs(has_detect_outdir)

    if not os.path.exists(lp_detect_outdir):
        os.makedirs(lp_detect_outdir)

    if not os.path.exists(lp_recognition_outdir):
        os.makedirs(lp_recognition_outdir)

    if not os.path.exists(helmet_recognition_outdir):
        os.makedirs(helmet_recognition_outdir)

    if not os.path.exists(lp_detect_bad_outdir):
        os.makedirs(lp_detect_bad_outdir)

    riding_classes = ['摩托车', '非机动']
    ids2ridingcls = dict(enumerate(riding_classes))
    ridingcls2ids = {c: idx for idx, c in ids2ridingcls.items()}

    test()
