import math
import argparse
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from lib.yolov3.darknet import Darknet
from lib.yolov3 import preprocess
from lib.yolov3.util import *

from lib.hrnet.lib.models import pose_hrnet
from lib.hrnet.lib.utils.transforms import transform_preds
from lib.hrnet.lib.config import cfg, update_config

def person_args():
    cur_dir = 'lib/yolov3'
    parser = argparse.ArgumentParser(description='YOLO v3 Cam Demo')
    parser.add_argument('--confidence', dest='confidence', type=float, default=0.70,
                        help='Object Confidence to filter predictions')
    parser.add_argument('--nms-thresh', dest='nms_thresh', type=float, default=0.4, help='NMS Threshold')
    parser.add_argument('--reso', dest='reso', default=416, type=int, help='Input resolution of the network. '
                                                                           'Increase to increase accuracy. Decrease to increase speed. (160, 416)')
    parser.add_argument('-wf', '--weight-file', type=str, default='lib/checkpoint/yolov3.weights', help='The path'
                                                                                                        'of model weight file')
    parser.add_argument('-cf', '--cfg-file', type=str, default=cur_dir + '/cfg/yolov3.cfg', help='weight file')
    parser.add_argument('-a', '--animation', action='store_true', help='output animation')
    parser.add_argument('-v', '--video', type=str, default='camera', help='The input video path')
    parser.add_argument('-np', '--num-person', type=int, default=1, help='number of estimated human poses. [1, 2]')
    parser.add_argument('--gpu', type=str, default='0', help='input video')

    return parser.parse_args()

def kps_args():
    cfg_dir = './lib/hrnet/experiments/'
    model_dir = './lib/checkpoint/'
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', type=str, default=cfg_dir + 'w48_384x288_adam_lr1e-3.yaml',
                        help='experiment configure file name')
    parser.add_argument('opts', nargs=argparse.REMAINDER, default=None,
                        help="Modify config options using the command-line")
    parser.add_argument('--modelDir', type=str, default=model_dir + 'pose_hrnet_w48_384x288.pth',
                        help='The model directory')
    parser.add_argument('--det-dim', type=int, default=416,
                        help='The input dimension of the detected image')
    parser.add_argument('--thred-score', type=float, default=0.30,
                        help='The threshold of object Confidence')
    parser.add_argument('-a', '--animation', action='store_true',
                        help='output animation')
    parser.add_argument('-np', '--num-person', type=int, default=1,
                        help='The maximum number of estimated poses')
    parser.add_argument("-v", "--video", type=str, default='camera',
                        help="input video file name")
    parser.add_argument('--gpu', type=str, default='0', help='input video')
    args = parser.parse_args()

    return args

def reset_config(args):
    update_config(cfg, args)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

def box_to_center_scale(box, model_image_width, model_image_height):
    """convert a box to center,scale information required for pose transformation
    Parameters
    ----------
    box : (x1, y1, x2, y2)
    model_image_width : int
    model_image_height : int

    Returns
    -------
    (numpy array, numpy array)
        Two numpy arrays, coordinates for the center of the box and the scale of the box
    """
    center = np.zeros((2), dtype=np.float32)
    x1, y1, x2, y2 = box[:4]
    box_width, box_height = x2 - x1, y2 - y1

    center[0] = x1 + box_width * 0.5
    center[1] = y1 + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_affine_transform(
        center, scale, rot, output_size,
        shift=np.array([0, 0], dtype=np.float32), inv=0
):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals

class Person_23d():
    def __init__(self):
        super(Person_23d, self).__init__()
        self.person_args = person_args()
        self.hr_args = kps_args()


    def reset_config(self):
        update_config(cfg, self.hr_args)

        # cudnn related setting
        cudnn.benchmark = cfg.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
        torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
        self.kps_args = cfg


    def load_model(self, CUDA=None, inp_dim=416):

        if CUDA is None:
            CUDA = torch.cuda.is_available()

        # Set up the neural network
        model = Darknet(self.person_args.cfg_file)
        model.load_weights(self.person_args.weight_file)
        # print("YOLOv3 network successfully loaded")

        model.net_info["height"] = inp_dim
        assert inp_dim % 32 == 0
        assert inp_dim > 32

        # If there's a GPU availible, put the model on GPU
        if CUDA:
            model.cuda()

        # Set the model in evaluation mode
        model.eval()
        self.person_model = model

    def yolo_human_det(self, img, reso=416, confidence=0.70):

        # args.reso = reso
        inp_dim = reso
        num_classes = 80

        CUDA = torch.cuda.is_available()

        img, ori_img, img_dim = preprocess.prep_image(img, inp_dim)
        img_dim = torch.FloatTensor(img_dim).repeat(1, 2)

        with torch.no_grad():
            if CUDA:
                img_dim = img_dim.cuda()
                img = img.cuda()
            output = self.person_model(img,CUDA)
            output = write_results(output, confidence, num_classes, nms=True, nms_conf=self.person_args.nms_thresh, det_hm=True)

            if len(output) == 0:
                return None, None

            img_dim = img_dim.repeat(output.size(0), 1)
            scaling_factor = torch.min(inp_dim / img_dim, 1)[0].view(-1, 1)

            output[:, [1, 3]] -= (inp_dim - scaling_factor * img_dim[:, 0].view(-1, 1)) / 2
            output[:, [2, 4]] -= (inp_dim - scaling_factor * img_dim[:, 1].view(-1, 1)) / 2
            output[:, 1:5] /= scaling_factor

            for i in range(output.shape[0]):
                output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, img_dim[i, 0])
                output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, img_dim[i, 1])

        bboxs = []
        scores = []
        for i in range(len(output)):
            item = output[i]
            bbox = item[1:5].cpu().numpy()
            # conver float32 to .2f data
            bbox = [round(i, 2) for i in list(bbox)]
            score = item[5].cpu().numpy()
            bboxs.append(bbox)
            scores.append(score)
        scores = np.expand_dims(np.array(scores), 1)
        bboxs = np.array(bboxs)

        return bboxs, scores

    def PreProcess(self,image, bbox):

        data_numpy = image


        c, s = box_to_center_scale(bbox, data_numpy.shape[0], data_numpy.shape[1])
        r = 0

        out_size = [256,256]
        trans = get_affine_transform(c, s, r, out_size)
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(256), int(256)),
            flags=cv2.INTER_LINEAR)

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
        input = transform(input).unsqueeze(0)
        return input, data_numpy, c, s

    # load model
    def model_load(self):
        model = pose_hrnet.get_pose_net(self.kps_args, is_train=False)
        if torch.cuda.is_available():
            model = model.cuda()

        state_dict = torch.load(self.kps_args.OUTPUT_DIR)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k  # remove module.
            #  print(name,'\t')
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        model.eval()
        # print('HRNet network successfully loaded')

        return model

    def get_final_preds(self, batch_heatmaps, center, scale):
        coords, maxvals = get_max_preds(batch_heatmaps)

        heatmap_height = batch_heatmaps.shape[2]
        heatmap_width = batch_heatmaps.shape[3]

        # post-processing
        if self.kps_args.TEST.POST_PROCESS:
            for n in range(coords.shape[0]):
                for p in range(coords.shape[1]):
                    hm = batch_heatmaps[n][p]
                    px = int(math.floor(coords[n][p][0] + 0.5))
                    py = int(math.floor(coords[n][p][1] + 0.5))
                    if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                        diff = np.array(
                            [
                                hm[py][px + 1] - hm[py][px - 1],
                                hm[py + 1][px] - hm[py - 1][px]
                            ]
                        )
                        coords[n][p] += np.sign(diff) * .25

        preds = coords.copy()

        # Transform back
        for i in range(coords.shape[0]):
            preds[i] = transform_preds(
                coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
            )

        return preds, maxvals