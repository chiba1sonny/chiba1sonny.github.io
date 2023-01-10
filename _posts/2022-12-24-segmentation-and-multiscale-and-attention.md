---
layout: post
title: Segmentation & Multi-scale & Attention
image: /img/cityscapes.png
tags: ["segmentation", "multi-scale", "attention"]
---

## 1.Image Segmentation
セグメンテーションは画像認識技術の一つで、画像をいくつかのオブジェクトに分割するタスクです。深層学習前の画像セグメンテーションの目的をSeparate image into coherent objects, Identify similar image regionsといえます。

<img width="200" alt="mojikyo45_640-2.gif" src="https://user-images.githubusercontent.com/68838083/211450349-62437e5b-e9e4-4d8b-b0a4-4dfc1b69ff9d.png">to
<img width="200" alt="mojikyo45_640-2.gif" src="https://user-images.githubusercontent.com/68838083/211450588-288b41af-21f0-4fb6-9a5e-b96ce5c88514.png">

深層学習時代からのセグメンテーションはこちらの３種類の手法に分けられます。それぞれはセマンティック・セグメンテーション、インスタンス・セグメンテーション、パノプティック・セグメンテーションです。セマンティックセグメンテーションは、画像中の全ての画素に対して、クラスラベルを予測することを目的とします。インスタンスセグメンテーションは、画像中の全ての物体に対して、クラスラベルを予測し、IDを付けることを目的とします。この2つのセグメンテーションを組み合わせたタスクはパノプティックセグメンテーションです。すなわち、画像中の全ての画素に対して、クラスラベルを予測し、IDを付けることを目的とします。

<img width="400" alt="mojikyo45_640-2.gif" src="https://user-images.githubusercontent.com/68838083/211450907-75b15d78-ce29-4e0d-a302-c2669485f1cf.png">

セグメンテーションは車の自動運転、衛星画像と医療画像解析、およびロボティックビジョンなどで活用されています。例えば、自動運転車の最も基本的なタスクは周囲の物体を瞬時に認識することです。セグメンテーションを活用することで、信号機、車両、車線などを正確に検知できます。

セグメンテーションアのプローチは、クラスタリング、CNN及びTransformerの三つの手法にたいべつされます。クラスタリングに、k-means, mean-shiftなどの方法があります。CNNに基づいたモデルとして、U-Net, DeepLabV3+などがあります。最近、SegFormer, TransUNetなどのTransformerに基づいた方法は流行っています。CNNに基づいたアプローチを簡単に紹介します。ネットワーク構造について、FCNを代表したFCN型、U-Netを代表したエンコーダ・デコーダ型、PSPNetを代表したピラミッド型とDeepLabシリーズの膨張畳み込みは主流です。
- FCN（Fully Convolutional Networks）

通常の画像分類CNNの畳み込み層の後は全結合層ですが、FCNでは全結合層を畳み込み層に置き換えます。そのため、Fully Convolutional Networksといえます。それで、最後の畳み込み層で得られたフィーチャーマップを入力画像と同じサイズまでアップサンプリングすることでピクセルごとのクラス予測を出力できます。それは、セマンティックセグメンテーション結果になります。

<img width="400" alt="mojikyo45_640-2.gif" src="https://user-images.githubusercontent.com/68838083/211451417-a9710c00-02dc-4cf5-9287-55247a9bf785.png">  <img width="50" alt="mojikyo45_640-2.gif" src="https://user-images.githubusercontent.com/68838083/211451494-e762b52a-c53c-4f56-bf28-488d8deb5d4c.png">  <img width="300" alt="mojikyo45_640-2.gif" src="https://user-images.githubusercontent.com/68838083/211451733-21409e04-8f4b-45bc-a4c0-a39459a8eb8e.png"> 

- U-Net

エンコーダ・デコーダ構造のモデルは、Encoderから特徴を抽出し、Decoderで確率マップを構成します。代表的なモデルはU-Netです。U-Netでは、エンコーダ・デコーダ構造以外、Skip connection技法を提案しました。U-Netの論文によると、Skip connectionより、エンコーダ各層での出力を、対応するデコーダ各層での入力に連結することで正確な位置情報を復元し、高解像度の出力を実現したと記載されています。このようなフィーチャーマップを連結する技法はPSPNet、DeepLabシリーズなどでも使われています。U-Netのエンコーダ・デコーダ構造はシンプルで精度も高いので、様々なセグメンテーションタスクに利用されています。Stable Diffusionでも使用されています。

<img width="400" alt="mojikyo45_640-2.gif" src="https://user-images.githubusercontent.com/68838083/211452030-cf9fa2e7-cb62-4be9-86db-33327d4b4786.png">

- PSPNet

PSP-netではピラミッド型プーリングが提案されました。異なるカーネルサイズでプーリングを行い、また畳み込みも行って、アップサンプリングしたものをプーリング前の特徴マップに連結します。このアプローチによって様々なスケールの画像の特徴を効率的に学習することを可能にしていますと論文に記載されています。

<img width="400" alt="mojikyo45_640-2.gif" src="https://user-images.githubusercontent.com/68838083/211452256-b9c002dd-2a82-4414-9642-d79519bf3bf6.png">

- DeepLabシリーズ

最後は、膨張畳み込みを提案したDeepLabシリーズです。膨張畳み込みは、カーネルの膨張により，受容野の範囲を拡大させて，より広い範囲の空間コンテキストを取得することができます。

<img width="60" alt="mojikyo45_640-2.gif" src="https://user-images.githubusercontent.com/68838083/211452370-ac9b23a1-fb4d-420c-9524-d59c9d750e6f.png"> <img width="80" alt="mojikyo45_640-2.gif" src="https://user-images.githubusercontent.com/68838083/211452442-1b63b31d-4b32-4d4f-9b16-2b8465e2d8bc.png"> <img width="100" alt="mojikyo45_640-2.gif" src="https://user-images.githubusercontent.com/68838083/211452460-669f22b0-fd57-4a7c-ad09-07d23f08e9cb.png">

- まとめ

CNNに基づいたセマンティックセグメンテーションのネットワーク構造をまとめました。こちらに示した4種類に分けられます。それぞれは、FCN型、エンコーダ-デコーダ、膨張畳み込み、ピラミッドプリングです。

<img width="200" alt="mojikyo45_640-2.gif" src="https://user-images.githubusercontent.com/68838083/211452873-2b36e054-5adc-4097-85b3-d2f95bce8898.png">




