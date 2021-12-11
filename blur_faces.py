#!/usr/bin/env python3
import sys
import logging
import argparse
from pathlib import Path
from typing import Tuple

from tqdm import tqdm
import cv2
from cv2 import CAP_PROP_FRAME_COUNT
import dlib
import numpy as np

logger = logging.getLogger(__name__)


def frame_iter(capture, description):
    def _iterator():
        while capture.grab():
            yield capture.retrieve()[1]
    return tqdm(
        _iterator(),
        desc=description,
        total=int(capture.get(CAP_PROP_FRAME_COUNT)),
    )


def blur(img: np.ndarray, rect: Tuple[int, int, int, int]) -> np.ndarray:
    """画像の指定位置にぼかしを作る

    Args:
        img (np.ndarray): 入力画像
        rect (Tuple[int, int, int, int]): 指定位置 [x1, y1, x2, y2]

    Returns:
        np.ndarray: ぼかしをいれた出力画像
    """
    (x1, y1, x2, y2) = rect
    i_rect = img[y1:y2, x1:x2]

    s_roi = cv2.blur(i_rect, (30, 30))  # ぼかし処理

    img2 = img.copy()
    img2[y1:y2, x1:x2] = s_roi

    return img2


def main(args):

    infile = Path(args.videofile)
    logger.debug(f"videofile: {infile}")
    if infile.exists() is False:
        logger.error(f"File is not exists: {infile}")
        sys.exit(1)

    video = cv2.VideoCapture(str(infile))

    face_detector = dlib.cnn_face_detection_model_v1(str(Path(__file__).resolve().parent / "mmod_human_face_detector.dat"))

    # 幅と高さを取得
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)
    logger.debug(f"width: {width}, height: {height}")

    # 総フレーム数を取得
    frame_rate = int(video.get(cv2.CAP_PROP_FPS))
    logger.debug(f"FPS: {frame_rate}")
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    logger.debug(f"output: {args.output}")
    writer = cv2.VideoWriter(args.output, fmt, frame_rate, size)

    for frame in frame_iter(video, args.output):
        # 顔検出
        faces = face_detector(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1)

        for i, f in enumerate(faces):
            x = f.rect.left()
            y = f.rect.top()
            w = f.rect.right() - x
            h = f.rect.bottom() - y
            frame = blur(frame,
                         (max(0, x), max(0, y), min(x+w, width), min(y+h, height)))

        writer.write(frame)

    # メモリの解放
    writer.release()
    video.release()


def parse_args():
    # オプションの解析
    parser = argparse.ArgumentParser(description='動画にぼかし処理をする')
    parser.add_argument(
                        'videofile',
                        )
    parser.add_argument(
                        '-o', '--output',
                        type=str,
                        default="output.mp4"
                        )
    parser.add_argument(
                        '-l', '--loglevel',
                        choices=('warning', 'debug', 'info'),
                        default='info'
                        )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    logger.setLevel(args.loglevel.upper())
    logger.info('loglevel: %s', args.loglevel)
    lformat = '%(name)s <L%(lineno)s> [%(levelname)s] %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=lformat,
    )
    logger.setLevel(args.loglevel.upper())

    main(args)
