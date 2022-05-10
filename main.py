import cv2
import torch
from common.model import *
from utils_all import Person_23d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from lib.preprocess import coco_h36m
import numpy as np
import argparse


l_leg = [[0,1],[1,2],[2,3]]
r_leg = [[0,4],[4,5],[5,6]]
bone = [[0,7],[7,8],[8,9],[9,10]]
l_hand = [[8,14],[14,15],[15,16]]
r_hand = [[8,11],[11,12],[12,13]]
rot = np.array([0.15232472, -0.1544232, -0.75475633, 0.619107], np.float32)
draw = [l_leg,r_leg,bone,l_hand,r_hand]

def args():

    parser = argparse.ArgumentParser(description='person3d args')
    parser.add_argument('--video', type=str, default=None, metavar='N', help='camera or video')
    parser.add_argument('--camera', type=int, default=0, metavar='C', help='camera or video')
    parser.add_argument('--arc', default=[3,3,3], metavar='LAYERS', help='filter widths separated by comma')
    parser.add_argument('--w', '--weight-file', type=str, default='./checkpoint/epoch_120_333.bin', help='The path')
    return parser.parse_args()


def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    qvec = q[..., 1:]
    uv = torch.cross(qvec, v, dim=len(q.shape) - 1)
    uuv = torch.cross(qvec, uv, dim=len(q.shape) - 1)
    return (v + 2 * (q[..., :1] * uv + uuv))

def camera_to_world(X, R, t):
    return wrap(qrot, np.tile(R, (*X.shape[:-1], 1)), X) + t


def wrap(func, *args, unsqueeze=False):
    """
    Wrap a torch function so it can be called with NumPy arrays.
    Input and return types are seamlessly converted.
    """

    # Convert input types where applicable
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)

    result = func(*args)

    # Convert output types where applicable
    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    else:
        return result



if __name__ == "__main__":
    args = args()

    if args.video != None:
        cap = cv2.VideoCapture(args.video)
    elif args.camera != None:
        cap = cv2.VideoCapture(args.camera)
    else:"camera or video load error,plese check your code !"

    w = 640
    h = 480
    cap.set(3, w)  # width=640
    cap.set(4, h)  # height=480




    p23d = Person_23d()
    p23d.load_model(CUDA=True)

    p23d.reset_config()
    pose_model = p23d.model_load()

    model = TemporalModel(17, 2, 17,filter_widths=args.arc, causal=True, dropout=0.25, channels=1024)
    checkpoint = torch.load(args.w, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['model_pos'])
    model = model.cuda()
    model.eval()


    kps_2d = []

    fig = plt.figure()



    while cap.isOpened():
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(45,45)
        radius = 3.0
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_zlim3d([0, radius / 2])
        ax.set_ylim3d([-radius / 2, radius / 2])


        _, frame = cap.read()
        b,c = p23d.yolo_human_det(frame)

        if b is None:
            continue

        cv2.rectangle(frame,(int(b[0][0]),int(b[0][1])),(int(b[0][2]),int(b[0][3])),(0,0,255),2)
        kps_input, data_numpy, center, scale = p23d.PreProcess(frame,b[0])
        kps_inputs = kps_input[:, [2, 1, 0]]

        #kps
        if torch.cuda.is_available():
            kps_inputs = kps_inputs.cuda()



        output = pose_model(kps_inputs)
        kps_pre,maxval = p23d.get_final_preds(output.clone().cpu().detach().numpy(), np.asarray([center]), np.asarray([scale]))

        h36m_kps,_ = coco_h36m(kps_pre)

        kps_2d.append(h36m_kps[0])
        fps_nums = 3**(len(args.arc))+1
        if len(kps_2d) == fps_nums:
            kps_2d.pop(0)
            input_2d = np.array([kps_2d])
            input_2d[:, :,:,0] /= w
            input_2d[:, :,:,1] /= h

            input_2d = torch.from_numpy(input_2d)
            if torch.cuda.is_available():
                input_2d = input_2d.cuda()
            pre_3d,pre_traj = model(input_2d)

            pre_3d = pre_3d.cpu().detach().numpy()
            pre_traj = pre_traj.cpu().detach().numpy()

            output = pre_3d + pre_traj


            output = camera_to_world(output, R=rot, t=0)

            x = output[:, :, :, 0]
            y = output[:, :, :, 1]
            z = output[:, :, :, 2]

            for dd in draw:
                for line in dd:
                    a = [x[0, 0, line[0]],x[0, 0, line[1]]]
                    b = [y[0, 0, line[0]],y[0, 0, line[1]]]
                    c = [z[0, 0, line[0]],z[0, 0, line[1]]]

                    ax.plot(a, b, c, c='r')



            ax.scatter(x, y, z)
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            plt.pause(0.001)
            plt.clf()


        for i in range(17):
            cv2.circle(frame,(int(h36m_kps[0][i][0]),int(h36m_kps[0][i][1])),2,(0,0,255),2)

        for dd in draw:
            for line in dd:
                cv2.line(frame,(int(h36m_kps[0][line[0]][0]),int(h36m_kps[0][line[0]][1])),
                (int(h36m_kps[0][line[1]][0]),int(h36m_kps[0][line[1]][1])),(0,0,255),2)

        cv2.namedWindow("show 2d keypoints",cv2.WINDOW_AUTOSIZE)
        cv2.imshow("show 2d keypoints", frame)
        cv2.waitKey(1)