import os
import cv2
import numpy as np
from scipy.stats import multivariate_normal

import torch
import torch.backends.cudnn as cudnn

from craft import CRAFT
import test
import imgproc
import file_utils


def generate_gt(net_pretrained, image, boxes, labels, args):
    region_gt = link_gt = np.zeros((image.shape[0], image.shape[1]), dtype='uint8')
    conf_map = np.zeros((image.shape[0], image.shape[1]), dtype='float32')
    gaussian = generate_gaussian(500)
    for i, box in enumerate(boxes):
        # Crop bounding box region
        warped = transform_image(image, box)

        # Apply pretrained network
        _, _, score_text, _ = test.test_net(net_pretrained, warped, args)

        # render results (optional)
        render_img = imgproc.cvt2HeatmapImg(score_text.copy())

        watershed = watershed_labeling(render_img)

        box_chr = chr_annotation(watershed)
        wordlen = len(labels[i])

        sconf = float(wordlen - min(wordlen, abs(wordlen - len(box_chr)))) / wordlen
        h, w = np.shape(score_text)
        if sconf < 0.5:
            box_chr = []
            bw = w // wordlen
            for j in range(wordlen):
                box_adj = np.array([[j * bw, 0], [(j + 1) * bw, 0], [(j + 1) * bw, h], [j * bw, h]])
                box_chr.append(box_adj)
            sconf = 0.5

        box_aff = []
        for k in range(len(box_chr) - 1):
            box_aff.append(get_affinity(box_chr[k], box_chr[k + 1]))

        conf_box = np.ones((h, w), dtype='float32') * sconf
        region_box = link_box = np.zeros((h, w), dtype='float32')

        for rbox in box_chr:
            region_box = restore(gaussian, region_box, rbox)

        for abox in box_aff:
            link_box = restore(gaussian, link_box, abox)

        region_gt = restore(region_box, region_gt, box)
        link_gt = restore(link_box, link_gt, box)
        conf_map = restore(conf_box, conf_map, box)

    return region_gt, link_gt, conf_map


def transform_image(image, pts):
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect = np.array([pts[np.argmin(s)],
                     pts[np.argmin(diff)],
                     pts[np.argmax(s)],
                     pts[np.argmax(diff)]], dtype='float32')
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def heatmap2score(image):
    # Create inverse map
    gray_values = np.arange(256, dtype=np.uint8)
    color_values = map(tuple, cv2.applyColorMap(gray_values, cv2.COLORMAP_JET).reshape(256, 3))
    color_to_gray_map = dict(zip(color_values, gray_values))

    # Convert image to BGR mapping
    image_copy = image.copy()

    # Return grayscaled map
    return np.apply_along_axis(lambda bgr: color_to_gray_map[tuple(bgr)], 2, image_copy)


def watershed_labeling(image):
    """
    :param image: score heatmap
    :return: watershed-applied image
    """
    # Convert to grayscale
    gray = heatmap2score(image)

    # Convert to Binary image
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphology opening, closing
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Dilate: get background
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Apply distance transform to get sure fg
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Subtract foreground from background
    unknown = cv2.subtract(sure_bg, sure_fg)

    # FG Labeling
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Apply watershed
    markers = cv2.watershed(image, markers)
    markers = np.abs(markers) - 1

    if np.max(markers) == 0:
        return markers.astype('uint8')

    markers_gray = np.multiply(markers, (255 // np.max(markers))).astype('uint8')
    return markers_gray


def chr_annotation(image):
    _, contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        box = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
        boxes.append(box)

    return boxes


def get_affinity(box1, box2):
    center_quad1 = np.average(box1, axis=0)
    center_quad2 = np.average(box2, axis=0)

    center_tri1_1 = np.average([box1[0], box1[1], center_quad1], axis=0).astype('int')
    center_tri1_2 = np.average([box1[2], box1[3], center_quad1], axis=0).astype('int')
    center_tri2_1 = np.average([box2[0], box2[1], center_quad2], axis=0).astype('int')
    center_tri2_2 = np.average([box2[2], box2[3], center_quad2], axis=0).astype('int')

    return np.array([center_tri1_1, center_tri2_1, center_tri2_2, center_tri1_2])


def restore(image, dst, pts):
    src_h, src_w = image.shape[:2]
    src_point = np.array([[0, 0], [src_w, 0], [src_w, src_h], [0, src_h]])

    h, status = cv2.findHomography(src_point, pts)

    if h is None:
        return dst

    restored = cv2.warpPerspective(image, h, (dst.shape[1], dst.shape[0]))

    return np.maximum(dst, restored)


def generate_gaussian(size):
    X = np.linspace(-1.35, 1.35, size)
    Y = np.linspace(-1.35, 1.35, size)
    X, Y = np.meshgrid(X, Y)

    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    F = multivariate_normal([0, 0], [[1, 0], [0, 1]])
    Z = F.pdf(pos)

    return Z / np.max(Z)


def ground_truth(args):
    # initiate pretrained network
    net = CRAFT()  # initialize

    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        net.load_state_dict(test.copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(test.copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    filelist, _, _ = file_utils.list_files('./data/train/data/IC15')

    for filename in filelist:
        # get datapath
        dataset = os.path.dirname(filename).split(os.sep)[-1]
        filenum = os.path.splitext(os.path.basename(filename))[0]
        label_dir = './data/train/ground_truth/{}/gt_{}/'.format(dataset, filenum)
        label_text = './data/train/ground_truth/{}/gt_{}.txt'.format(dataset, filenum)

        if not os.path.exists(label_dir):
            os.mkdir(label_dir)

        image = imgproc.loadImage(filename)

        gt_boxes = []
        gt_words = []
        with open(label_text, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
            for line in lines:
                gt_data = line.strip().split(',')
                gt_box = gt_data[:8]
                if len(gt_data) > 9:
                    gt_word = ','.join(gt_data[8:])
                else:
                    gt_word = gt_data[-1]
                gt_box = [int(a) for a in gt_box]
                gt_box = np.reshape(np.array(gt_box), (4, 2))
                gt_boxes.append(gt_box)
                gt_words.append(gt_word)

        score_region, score_link, conf_map = generate_gt(net, image, gt_boxes, gt_words, args)

        torch.save(score_region, label_dir + 'region.pt')
        torch.save(score_link, label_dir + 'link.pt')
        torch.save(conf_map, label_dir + 'conf.pt')





