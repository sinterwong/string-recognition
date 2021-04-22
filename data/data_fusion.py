import cv2
import glob
import os.path as osp
import os
import numpy as np
import random


def add_haze(image, t=0.6, A=1):
    '''
        添加雾霾
        t : 透视率 0~1
        A : 大气光照
    '''
    out = image * random.uniform(t, 1.0) + A * 255 * (1-t)
    return out


def ajust_image(image, cont=1, bright=0):
    '''
        调整对比度与亮度
        cont : 对比度，调节对比度应该与亮度同时调节
        bright : 亮度
    '''
    out = np.uint8(np.clip((cont * image + bright), 0, 255))
    # tmp = np.hstack((img, res))  # 两张图片横向合并（便于对比显示）
    return out


def ajust_jpg_quality(image, q=100, save_path=None):
    '''
        调整图像JPG压缩失真程度
        q : 压缩质量 0~100
    '''
    cv2.imwrite("jpg_tmp.jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    out = cv2.imread('jpg_tmp.jpg')
    return out


def add_gasuss_blur(image, kernel_size=(3, 3), sigma=0.1):
    '''
        添加高斯模糊
        kernel_size : 模糊核大小
        sigma : 标准差
    '''
    out = cv2.GaussianBlur(image, kernel_size, sigma)
    return out


def data_processing(plate_img, src_im_p):
    ph, pw, _ = plate_img.shape
    plate_ploy = np.float32([[pw, ph], [0, ph], [0, 0], [pw, 0]])
    plate_img[plate_img == 0] += 1   # 为了之后过滤黑边

    src_ploy = np.float32([list(map(int, b.split('&'))) for b in src_im_p.split('-')[4].split('_')])

    # 排序 x
    sort_x_ids = np.argsort(src_ploy[:, 0])
    left_coord = src_ploy[sort_x_ids[:2]]
    right_coord = src_ploy[sort_x_ids[2:]]

    # 取前两个最小的, 再排序 y, 取这两个最小的 作为左上, 另一个为 左下, 右边同样
    left_coord = left_coord[np.argsort(left_coord[:, 1])]  # get left up, left down
    right_coord = right_coord[np.argsort(right_coord[:, 1])]  # get right up, right down

    # 右下，左下，左上，右上
    src_ploy = np.array([right_coord[1], left_coord[1], left_coord[0], right_coord[0]]).astype(np.float32)
    src_bbox = [list(map(int, b.split('&'))) for b in src_im_p.split('-')[3].split('_')]
    sw = src_bbox[1][0] - src_bbox[0][0]
    sh = src_bbox[1][1] - src_bbox[0][1]

    # 将src_ploy 归到原点
    src_ploy[:, 0] -= np.min(src_ploy[:, 0])
    src_ploy[:, 1] -= np.min(src_ploy[:, 1])

    M = cv2.getPerspectiveTransform(plate_ploy, src_ploy)
    plate_dst = cv2.warpPerspective(plate_img, M, (sw, sh))

    img = cv2.imread(src_im_p)
    # src_img = img.copy()

    # 过滤黑边覆盖
    img[src_bbox[0][1]: src_bbox[1][1], src_bbox[0][0]: src_bbox[1][0]][plate_dst > 0] = plate_dst[plate_dst > 0]
    
    # cv2.imwrite('dst.jpg', plate_dst)
    # cv2.imwrite('plate_img.jpg', plate_img)
    # cv2.imwrite('result.jpg', img)
    # cv2.imwrite('src.jpg', src_img)

    if random.random() > 0.85:
        img = add_haze(img)
    # if random.random() > 0.6:
    #     img = ajust_image(img)
    # if random.random() > 0.6:
    #     img = ajust_jpg_quality(img)
    if random.random() > 0.7:
        img = add_gasuss_blur(img)

    plate_result = img[src_bbox[0][1]: src_bbox[1][1], src_bbox[0][0]: src_bbox[1][0]]

    return img, src_bbox[0] + src_bbox[1], plate_result


def main():

    if not osp.exists(out_root):
        os.makedirs(out_root)

    if not osp.exists(plate_out_root):
        os.makedirs(plate_out_root)

    src_list = glob.glob(src_root + '/*.jp*')
    plate_list = glob.glob(plate_root + '/*.jp*')
    # plate_list = plate_list + plate_list + plate_list

    for i, im_p in enumerate(plate_list):
        plate_img = cv2.imread(im_p)
        # plate = osp.basename(im_p).split('_')[0]  # 原始
        plate = osp.basename(im_p).split('_')[1].replace('.jpg', '')  # 新能源
        src_im_p = random.choice(src_list)
        result, bbox, plate_dst = data_processing(plate_img, src_im_p)

        # result_name = '%s_double_%05d.jpg' % (plate, i)
        result_name = '%s_NE_%05d.jpg' % (plate, i)
        bbox = ','.join(list(map(str, bbox)))

        cv2.imwrite(osp.join(out_root, result_name), result)
        cv2.imwrite(osp.join(plate_out_root, plate + '_GE_%d.jpg' % i), plate_dst)
        with open(osp.join(out_root, result_name.replace('jpg', 'txt')), mode='w') as wf:
            wf.write('{}\t{}\t{}\t{}'.format(result_name, 1, bbox, plate))


if __name__ == '__main__':
    src_root = '/home/wangjq/license_plate_dataset/licence-plate/detect_val'
    plate_root = '/home/wangjq/license_plate_dataset/licence-plate/src_GE_train'
    out_root = 'detect_NE_train'
    plate_out_root = 'plate_NE_train'
    main()

