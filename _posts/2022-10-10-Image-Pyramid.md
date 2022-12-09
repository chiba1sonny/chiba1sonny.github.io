---
layout: post
title: Image Pyramid (画像ピラミッド)と深層学習中の応用
image: /img/pyramid.png
tags: ["image analysis", "deep learning"]
---

## 1.Image pyramidとは
Image pyramidは基本的に、pyramidの形のような解像度が異なる一連の画像を言います。元画像をダウンサンプリングすると、低解像度の画像が生成されます（アップサンプリングで、高解像度の画像が生成されます）。何回繰り返したら、解像度が異なる一セットの画像が得られます。ダウンサンプリングする方法には、Gaussian kernelという画像フィルタリング方法と、Bilinear-InterprolationというInterprolation方法などあります（アップサンプリングならLaplacian kernelでできます）。こうやって得られた一連の画像は、トップの低解像度からボトムの高解像度まで積み重ねられ、pyramidの形に似るので、Image pyramidと呼ばれます。

![image](https://user-images.githubusercontent.com/68838083/197445960-bee88ff3-a84d-42e2-9a6f-d032d1c414ff.png)

## 2.なぜImage pyramidを使うの
- 実は、コンピュータービジョンや深層学習に関して、多くの優れたアイディアやネットワーク構造は、人間の脳と視覚の仕組みを模倣することで生成されたんです。Image pyramidがその一例です。人間は画像を観察する場合、画像からの距離が長くなると（画像サイズが小さくなることに相当）、画像中の小さなオブジェクトは見えなくなリます。逆に、画像中の大きなオブジェクトはある程度の範囲でも見えます。そのため、大きなオブジェクトは、低解像度の画像でも簡単に認識されますが、小さなオブジェクトを精確に認識する場合、高解像度の画像を利用すべきです。
- Image pyramidのもう一つ利点は、coarse to fine戦略を使用して物体検出を高速化できることです。
- また、セマンティックセグメンテーションについて、Image pyramidで入力画像を設置して利用することで、高解像度の画像でfine detailなど小さなオブジェクトをよく分割し、低解像度の画像で大きなオブジェクトをよく分割することができます。

## 3.Image pyramidを利用したアイディア
- DeepLabの著者のLiang-Chieh Chenさんの論文「[Attention to Scale](https://openaccess.thecvf.com/content_cvpr_2016/papers/Chen_Attention_to_Scale_CVPR_2016_paper.pdf)」では、元画像を３つや２つのスケールに変更し、その一連の画像を入力としてニューラルネットワークに通過させました。結果として、このようなマルチスケールの入力（実はImage pyramid）がシングルスケールの入力より、セマンティックセグメンテーションのパフォーマンスを向上させました。

![Screenshot from 2022-10-24 13-36-18](https://user-images.githubusercontent.com/68838083/197449371-e877c51d-80af-455d-b8bc-a2180c743c33.png)

- NVIDIAの研究「[HIERARCHICAL MULTI-SCALE ATTENTION FOR SEMANTIC SEGMENTATION](https://arxiv.org/pdf/2005.10821.pdf)」では、fine detailなど小さなオブジェクトは、拡張された画像でよくセグメンテーションされ、逆に大きなオブジェクトは、縮小された画像でよく分割されことを証明しました。

![Screenshot from 2022-10-24 13-43-33](https://user-images.githubusercontent.com/68838083/197450007-a1e40d27-aecd-412a-8a6c-66b60c9d01ba.png)

- ResNetの著者のKaiming Heさんは、下の図のようなpooling strategyを利用することで、ネットワークの入力サイズに関係なく、サイズが固定された特徴ベクトルを生成させました。基本的には、あるサイズの入力が与えられると、３つのMax poolingは同時に実行され、それぞれから4 x 4、2 x 2、1 x 1の特徴マップが生成されます。次に、これらを線形化してから１つのベクトルを取得します。論文によると、このベクトルには、コンピュータービジョンタスクの精度を向上させできる空間情報が含まれています。これは有名な「[Spatial pyramid pooling](https://arxiv.org/abs/1406.4729)」です。前の２つの例は入力画像のImage pyramidを使用したと言うと、Spatial pyramid poolingは特徴マップのImage pyramidを利用したと言えます。セマンティックセグメンテーションタスクのPSPNetのPyramid pooling module, DeeplabのAtrous spatial pyramid pooling(ASPP)などもSPPのアイディアからインピレーションを得ました。

![Screenshot from 2022-10-24 13-59-20](https://user-images.githubusercontent.com/68838083/197451589-4d904c31-6afb-453c-a1fa-c49ec76ba4cc.png)

## 4.OpenCvの関数でImage pyramidを実現してみる
- Image pyramidを実現するには、２つの路線があります。１つ目は、下から上へ、すなわち、高解像度から低解像度へ（低いレベルから高いレベルへ）です。OpenCvの```cv2.pyrDown```関数を使い
ます。Gaussian kernelを利用し、基本的には、```i```層から```i+1```層を生成します。まず、```i```層をGaussian kernelで畳み込み操作を行い、次に偶数の行と列をすべて削除します。こう
すると、新たに生成された画像のwidthとheightはもと画像の１/２になり、サイズは１/４になります。このプロセスを繰り返し、Image pyramidが得られます。

```py
import cv2
moto = cv2.imread('icon.jpg')
hiku = cv2.pyrDown(moto)
sai_hiku = cv2.pyrDown(hiku)
cv2.imshow('image', moto)
cv2.imshow('image', hiku)
cv2.imshow('image', sai_hiku)
```

![image](https://user-images.githubusercontent.com/68838083/197453008-b9fedaa5-9ba9-46ad-a785-1e1c4aea4cb8.png)

![image](https://user-images.githubusercontent.com/68838083/197452858-29169a7f-0490-4245-b287-d25c9271ba71.png)

![image](https://user-images.githubusercontent.com/68838083/197453151-53802496-7520-4ba2-a8e4-313dbc0a1c1c.png)

- ２つ目は、Laplacian kernelを用いたアップサンプリングです。やり方は、画像の各チャネルでサイズを２倍にして、新たに生成されたところでは０を埋めます。こうやると、画像が圧縮されたので、質が悪くなります。使った関数は```cv2.pyrUp```です。以下のコードで、前の元画像をダウンサンプリングして得られた```hiku画像```をアップサンプリングしました。この結果を元画像と比較してみてください。

```py
taka_from_hiku = cv2.pyrUp(hiku)
cv2.imshow('image', taka_from_hiku)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

![image](https://user-images.githubusercontent.com/68838083/197456630-2d75a0c5-3073-4c91-a96f-bbf8ada69576.png)
