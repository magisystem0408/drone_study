## ハンドジェスチャーによるIOTデバイスの遠隔コントロールの検討

## 機能
- 音楽モード
  - 音楽を再生する(BAG)
  - 音楽を停止する(BAG)
  - ボリューム調整モードに入る
    - ボリュームを調整
    - ボリューム戻る
- ライトモード
  - ライトをつける
  - ライトを消す
  - ライト明るさを調整する
    - ライトを調整する
    - ライト調整モードを抜ける
    
- 実装したもの
  - 簡単なモード制御
- 実装時に発生しているバグと対処法
  - 音楽が何回も流れてしまう
    - whileのなかで1回だけ実行できるようにする。

- もっとできる場合
  - 今のコードに左右の手を認識する
  - classで描き直す(ラグが出ることがあるので)

- 考えて欲しいものは
  - 他の人がやると認識精度が悪くなる。
  - 