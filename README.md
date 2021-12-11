# blur_faces

```
先端系課題[001]

テーマ名：人物匿名化処理の実装
課題内容：
動画を入力にして
顔のぼかしを施した動画がアウトプットされるシステムの構築
```

## インストール

Ubuntu 20.4LTS Python 3.7 で動作確認

```
$ pip install -r requirements.txt
```

## 使いかた

```
$ python blur_faces.py -h
usage: blur_faces.py [-h] [-o OUTPUT] [-l {warning,debug,info}] videofile

動画にぼかし処理をする

positional arguments:
  videofile

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
  -l {warning,debug,info}, --loglevel {warning,debug,info}
```

## 使用したシステムの解説

### 1. 動画の読み書き、ぼかし処理

OpenCVを使用(細かい解説は割愛)

### 2. 顔の検知

Convolutional Neural Network (CNN) による顔検出を [Dlib で公開されている学習済みモデル mmod_human_face_detector.dat](https://github.com/davisking/dlib-models#mmod_human_face_detectordatbz2)を使って行った

動画から抽出したフレーム画像に顔が検知された場合、バウンディングボックスの座標が取得できるので、そこを上記のOpenCVのblur()関数を使用してぼかし処理している


## 制限

1. 80x80ピクセル以下の「小さな」顔にはぼかしがかからない
2. 出力された動画に音声情報は含まれていない
