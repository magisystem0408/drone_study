#　3次元姿勢推定技術(openpose media-pipe)を使用してジェスチャーでドローンを操作を行う。
---
- ドローンはDjI社製のドローンを使用する。

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
- 実験その5
  - 動画はまだとってない
    - L字を左右で区別し、ポーズすると空中でドローンが前転。だたしこれは離陸している時のみのコマンドである。
- 実験その8
  - https://youtu.be/lU0TZhb3IEM
    - 仮想空間上で動かせるようにした。
## ファイル構成
- pose_manager.py
  - mediapipeでジェスチャー認識のみ入ってる
- tello_manager.py
  - ジェスチャーとドローンを動かせる


## 次やろうと思うこと
- コマンドがいっぱい送られてきた時、一つを実行しているとは、後からきたやつをむしするblockingIOを導入する。このためにはtheadingを開く必要がある

- 顔追従をできるようにする。
- ドローンカメラをURLで受け取ると遅いのでffmpegで変換してリアルタイムに表示できるようにする
- ジェスチャーのパターンを増やしてみでドローンを操作できるようにする
  - pose+handに移行させる
    - 事前テスト実証
      - ETみたいに左手と右手の人差し指をくっつけたあとピンチアウトするとTrueまたはその距離を図ってドローンにコマンドを送る処理

- ドローンのカメラを使用してジェスチャーを可能にする


## 参考文献等
- https://spout.zeal.co
- https://github.com/dji-sdk/Tello-Python
- https://github.com/CMU-Perceptual-Computing-Lab/openpose
- https://google.github.io/mediapipe/



