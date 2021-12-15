import cv2
import numpy as np
import torch
from detected_Face.functions.prior_box import PriorBox
from detected_Face.Config.config import cfg_mnet
from detected_Face.utils.box_utils import decode, decode_landm
from detected_Face.utils.nms.py_cpu_nms import py_cpu_nms
from detected_Face.models.retinaface import RetinaFace
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cpu', type=str, default='cpu', help='Use cpu inference')
parser.add_argument('--img_path', type=str, default="./data/tuankien3.jpg", help='path of file img')
parser.add_argument('--weight', type=str, default='./weights/mobilenet0.25_Final.pth', help='path of file weight')
args = parser.parse_args()


class Detect:

    def __init__(self, img_path, device, weight):
        self.cfg = Detect.load_config(self=self)
        self.weight = weight
        self.img_path = img_path
        self.device = device
        self.img, self.im_height, self.im_width, self.scale = Detect.scale_img(self=self, img_path=img_path,
                                                                               device=self.device)
        self.net = Detect.load_model(self=self, weight=self.weight, cfg=self.cfg)
        self.dets, self.landms = Detect.detect(self=self, net=self.net, img=self.img, scale=self.scale,
                                               device=self.device, cfg=self.cfg,
                                               im_height=self.im_height, im_width=self.im_width)

    # load Config
    def load_config(self):
        self.cfg = cfg_mnet
        return self.cfg

    def scale_img(self, img_path, device):
        self.img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        self.img = np.float32(self.img)
        im_height, im_width, _ = self.img.shape
        self.scale = torch.Tensor([self.img.shape[1], self.img.shape[0], self.img.shape[1], self.img.shape[0]])
        self.img -= (104, 117, 123)
        self.img = self.img.transpose(2, 0, 1)
        self.img = torch.from_numpy(self.img).unsqueeze(0)
        self.img = self.img.to(device)
        return self.img, im_height, im_width, self.scale

    # load model
    def load_model(self, weight, cfg):
        self.net = RetinaFace(cfg=cfg, phase='test')
        self.net.load_state_dict(torch.load(weight, map_location="cpu"))
        self.net.eval()
        return self.net

    # detect
    def detect(self, net, img, device, cfg, im_height, im_width, scale):
        scale = scale.to(device)
        resize = 1
        loc, conf, self.landms = net(img)
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        self.landms = decode_landm(self.landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        self.landms = self.landms * scale1 / resize
        self.landms = self.landms.cpu().numpy()
        # ignore low scores
        inds = np.where(scores > 0.02)[0]  # confidence_threshold
        boxes = boxes[inds]
        self.landms = self.landms[inds]
        scores = scores[inds]
        # keep top-K before NMS
        order = scores.argsort()[::-1][:5000]  # --top_k
        boxes = boxes[order]
        self.landms = self.landms[order]
        scores = scores[order]

        # do NMS
        self.dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(self.dets, 0.4)  # nms_threshold
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        self.dets = self.dets[keep, :]
        self.landms = self.landms[keep]

        # keep top-K faster NMS
        self.dets = self.dets[:750, :]  # keep_top_k
        self.landms = self.landms[:750, :]
        return self.dets, self.landms


if __name__ == "__main__":
    detect = Detect(args.img_path, args.cpu, args.weight)
    dets = detect.dets
    lam = detect.landms

