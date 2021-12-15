import argparse
import numpy as np

import arc_Face.arcface
import detected_Face.detect
import cv2
from detected_Face import *

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cpu', help='Use cpu inference')
parser.add_argument('--img_path', type=str, default="./detected_Face/data/tuankien3.jpg", help='path of file img')
parser.add_argument('--weight_detect', type=str, default='./detected_Face/weights/mobilenet0.25_Final.pth',
                    help='path of '
                         'file '
                         'weight')
parser.add_argument('--weight_arcFace', type=str, default='./arc_Face/weights/backbone167.pth', help='path of '
                                                                                                     'file '
                                                                                                     'weight')
args = parser.parse_args()

if __name__ == "__main__":
    # weight path
    weight_arc = args.weight_arcFace
    weight_detect = args.weight_detect

    # detect face
    dect = detected_Face.detect.Detect(args.img_path, args.device, weight_detect)
    detect_box = dect.dets
    detect_box = np.concatenate(detect_box, axis=0)
    b = list(map(int, detect_box))

    img_raw = cv2.imread(args.img_path, cv2.IMREAD_COLOR)
    img_crop = img_raw[b[1]:b[3], b[0]:b[2], :]
    cv2.imwrite('./data_face_detected/face_tuankien3.jpg', img_crop)

    arcface = arc_Face.arcface.Arcface('./data_face_detected/face_tuankien3.jpg', weight_arc, args.device)
    feat = arcface.feat

    print(feat)
