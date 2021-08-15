##　3次元姿勢推定技術(openpose media-pipe)を使用してジェスチャーでドローンを操作を行う。

---
- ドローンはDjI社製のドローンを使用する。


- 最新実験はこちら(実験4)
    - https://youtu.be/z29qPcBg7uI

## 過去の実験動画
- 実験その1
    - https://youtu.be/mtdKMVZsfZ0
      - Spout(参考文献参照)を使用してUnityのメインカメラをwebカメラとして
    認識して排出し、openposeをpython側で読み込みwebカメラ情報を読み込んで認識させる
- 実験その2
    - https://youtu.be/QuG8XfYNUTE
      - openposeをインストールしてwebカメラで大勢がうつせるかを確かめてみた
- 実験その3
    - https://youtu.be/TKZqfSrrq1Q
        - port:11111からUDPで送られてくるドローンからうつされたカメラ情報を取得し
  、顔、目の検出をした。
      
- 実験その4
    - https://youtu.be/z29qPcBg7uI
      - webカメラであるがポーズを認識して離陸、着陸できるようになった。
  T字ポーズで離陸。離陸中に両手をあげると着陸する。ドローンのカメラで認識したかったがドローンから送られてくる映像が遅すぎて先に、パソコンのwebカメラを使用してジェスチャーで離陸させられるようにした
        
## ファイル構成
- pose_manager.py
  - mediapipeでジェスチャー認識のみ入ってる
- tello_manager.py
  - ジェスチャーとドローンを動かせる

## 参考文献等
- https://spout.zeal.co
- https://github.com/dji-sdk/Tello-Python
- https://github.com/CMU-Perceptual-Computing-Lab/openpose
- https://google.github.io/mediapipe/



