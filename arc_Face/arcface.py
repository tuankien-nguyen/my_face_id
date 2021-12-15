import argparse
import torch
import cv2
import numpy as np


from arc_Face.backbones import get_model
from arc_Face.Config.config import cfg_arcface
import torch.nn.functional as F

from numpy.linalg import norm

parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type=str, default="../detected_Face/face0.jpg", help='path of file img')
parser.add_argument('--device', type=str, default='cpu', help='Use cpu inference')
parser.add_argument('--weight', type=str, default='./weights/backbone167.pth', help='path of file weight')
args = parser.parse_args()


def normalize(embedding):
    embedding_norm = norm(embedding)
    normed_embedding = embedding / embedding_norm
    return normed_embedding


class Arcface:
    def __init__(self, img_path, weight, device):
        self.conf = Arcface.load_config(self=self)
        self.img_path = img_path
        self.weight = weight
        self.device = device
        self.img = Arcface.scale_img(self=self, img_path=img_path)  # self.load_config
        self.net = Arcface.load_model(self=self, conf=self.conf, weight=self.weight, device=self.device)
        self.feat = Arcface.inference(self=self, net=self.net, img=self.img)

    def load_config(self):
        self.conf = cfg_arcface
        return self.conf

    def scale_img(self, img_path):
        self.img = cv2.imread(img_path)
        self.img = cv2.resize(self.img, (112, 112))
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.img = np.transpose(self.img, (2, 0, 1))
        self.img = torch.from_numpy(self.img).unsqueeze(0).float()
        self.img.div_(255).sub_(0.5).div_(0.5)
        return self.img

    # load model
    def load_model(self, conf, weight, device):
        self.net = get_model(conf['model'], fp16=False)
        self.net.load_state_dict(torch.load(weight, map_location=torch.device(device)))
        self.net.eval()
        return self.net

    # inference
    def inference(self, net, img):
        self.feat = net(img).detach().numpy()
        return self.feat


if __name__ == "__main__":
    img_path = '../data_face_detected/face_img_tuankien1.jpg'
    img_path2 = '../data_face_detected/face_tuankien3.jpg'
    arc1 = Arcface(img_path, args.weight, args.device)
    arc2 = Arcface(img_path2, args.weight, args.device)
    a = arc1.feat.tolist()
    b = arc2.feat.tolist()
    norm_a = normalize(a[0])
    norm_b = normalize(b[0])
    a = torch.FloatTensor(norm_a)
    b = torch.FloatTensor(norm_b)
    result = F.cosine_similarity(a, b, dim=0)
    print(float(result))
