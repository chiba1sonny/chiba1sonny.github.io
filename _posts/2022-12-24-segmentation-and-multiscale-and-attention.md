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

<img width="600" alt="mojikyo45_640-2.gif" src="https://user-images.githubusercontent.com/68838083/211452873-2b36e054-5adc-4097-85b3-d2f95bce8898.png">

## 2.セグメンテーションに関するマルチスケール・アテンション
続いて、セグメンテーションに関するマルチスケールです。マルチスケール技法を二つに分けられます。それぞれは、入力画像におけるマルチスケールとフィーチャーマップにおけるマルチスケールです。

- マルチスケール技法
##### 1.入力画像におけるマルチスケール
下の図のように、入力画像におけるマルチスケールは、解像度が異なる複数の入力画像を使用して、各スケールから得られた特徴を混（ま）ぜ合わせて予測精度を向上させます。画像ピラミッドとも言えます。

<img width="200" alt="mojikyo45_640-2.gif" src="https://user-images.githubusercontent.com/68838083/211453463-13fc8e4d-844e-4eae-b443-8f6635807eee.png">

##### 2.フィーチャーマップにおけるマルチスケール
PSPNetはフィーチャーマップにおけるマルチスケールの代表例です。フィーチャーマップをピラミッドの形で伝播させていくことで、様々なスケールの画像の特徴を効率的に学習できます。1の入力画像におけるマルチスケールと比べると計算量も少ないです。

- 入力画像におけるマルチスケール

なぜマルチスケールの画像を入力とするかと言うと、マルチスケールオブジェクト問題を解決できるからです。画像データにおいて、検出対象の物体の大きさは大小様々です。そのため、画像のスケールを変えて複数回推論し、結果を統合するというアプローチは深層学習が登場する前からありました。こちらの図を見ると、大きな物体（道路）は小さいスケール（0.5x Scale）では、良くセグメンテーションされました。小さい物体（柱）は大きいスケール（2.0x Scale）では、良くセグメンテーションされました。もし、その二つの結果を統合すると良い結果が得られるかもしれません。これは、マルチスケール技法です。

<img width="300" alt="mojikyo45_640-2.gif" src="https://user-images.githubusercontent.com/68838083/211454134-a26ab870-9f2f-4665-9341-a04f3e81468b.png">

- 入力画像におけるマルチスケール：各スケールの予測をマージする方法

統合する方法、すなわち、各スケールの予測をマージする方法として、平均プーリングとマックスプーリングなどあります。下の表を見ると、マルチスケール入力はシングル入力の予測結果を向上させました。

<img width="300" alt="mojikyo45_640-2.gif" src="https://user-images.githubusercontent.com/68838083/211454340-e6a55b8e-4628-44de-8eac-314e6ffd922c.png">

それで、各スケールの予測のマージについて、平均プーリングとマックスプーリングよりも良い方法を提案した「Attention to scale」が登場しました。これは2017年にCVPRで発表された論文です。第一著者はDeepLabシリーズの著者です。この論文では、平均プーリングとマックスプーリングをAttentionメカニズムに替えて、各スケールにおける予測をマージしました。下はAttention to scaleのモデル図です。Attentionメカニズムは、画像データのどこに注目すべきかを学習させる仕組みであり、深層学習全般（ぜんぱん）で用いられます。このAttentionメカニズムがコンピュータビジョンにおいて用いられた場合は、特徴マップのどこに注目すべきかを学習させることになります。そして、得られたAttention値を重みとして使い、各スケールから得られた特徴を混ぜ合わせて，調和的な予測ができます。下の表はPASCALデータセットにおける各入力方法の結果です。Attentionメカニズムを使用することで、最高の精度が得られました。

<img width="300" alt="mojikyo45_640-2.gif" src="https://user-images.githubusercontent.com/68838083/211454533-97dcc9b0-f96d-4455-b6b2-b049af96cc81.png">　<img width="300" alt="mojikyo45_640-2.gif" src="https://user-images.githubusercontent.com/68838083/211454630-8b5e1a4f-9540-4f23-b901-77eadfd144e0.png">

しかし、この論文の提案手法は各スケールごとにAttention値を算出しているため、メモリー消費量が多い、トレーニング時間が長いといった欠点が存在します。

<img width="300" alt="mojikyo45_640-2.gif" src="https://user-images.githubusercontent.com/68838083/211454783-67ccca5b-8336-4a20-9fb2-a294b598ae61.png">

この欠点に対して、論文Hierarchical multi-scale attention for semantic segmentationでは、Attention値は隣接するスケール間で学習される手法を提案しました。そして、推論を行う時、トレーニングと異なるマルチスケール入力を選択することは可能になりました。トレーニング時の計算量と時間を減らして、精度も向上させました。

<img width="300" alt="mojikyo45_640-2.gif" src="https://user-images.githubusercontent.com/68838083/211454897-401721fb-980e-4884-b800-42d8e34c27cd.png">

提案手法によるトレーニングプロセスを説明します。左側の入力画像についてですが、スケール１とスケール２の二つの入力画像があります。まず、Scale1とScale2の2つスケールの画像を入力とし、ネットワークに通過させます。ちゃいろのTrunkでは、特徴抽出を行います。得られたフィーチャーマップを緑のアテンションモジュールとセマンティックヘードに通過させ、それぞれからアテンション値と各スケールによる予測結果を生成します。最後、二つのスケールによる予測結果をAttention値に基づき、マージして最後の予測画像を取得します。Attention値は隣接するスケール間で学習されるため、推論を行う時、トレーニングと異なるマルチスケール入力を選択できます。

そこで、推論プロセスを三つのスケールの画像を入力とした例で説明します。こちらの例で、スケール1、スケール2、スケール3の画像を入力とした推論プロセス図を示します。トレーニングと同じように、特徴抽出、Attention値と各スケールによる予測は生成された後、三つのスケールによる予測をマージしてから最終予測を取得します。提案手法は、各スケールの利点を組み合わせ、調和的な予測を行うことができます。

こちらに、DeepLab V3+とMapillaryデータセットを用いた各方法の比較を示します。結果として、本論文の手法は最高の精度を得て、計算量とトレーニング時間も先行研究より少ないことは分かります。

Method:入力方法（Single Scale：シングル入力；AvgPool：マルチスケール入力における結果を平均プーリングでマージする；Explict：論文「Attention to Scale」；Hierarchical：論文「Hierarchical」．

Eval scales：推論に使うスケールセット

IOU：Intersection over Union；大きいほど精度が高い．

FLOPS：トレーニング時の計算量；少ないと良い．

Minibatch training time：minibatchを学習される時間（Nvidia Tesla V 100 GPU）；少ないと良い．

<img width="400" alt="mojikyo45_640-2.gif" src="https://user-images.githubusercontent.com/68838083/211455446-d235b596-6832-400f-8310-37d955e4116d.png">

この論文では、マルチスケール入力におけるAttentionメカニズムを提案したので、Attentionメカニズムについての考察を行って、提案手法の有効性を検証しました。上の二つは元画像です。下のは、各スケールにおける予測とAttention値をマップした図です。Attention値は、画像の位置ごとに注目している値であり、白い色は高いアテンション値を示し、黒い色なら低いアテンション値を示します。図を見ると、小さい物体（はしら）は大きいスケール（2.0x Scale）では、良くセグメンテーションされました。それらのAttention値も高いです。
　大きなもの（道路）は小さいスケール（0.5x Scale）では、良くセグメンテーションされました。それのAttention値も高いです。

<img width="400" alt="mojikyo45_640-2.gif" src="https://user-images.githubusercontent.com/68838083/211455802-c3ed4f62-b67f-48c6-a2cc-a151fb7ecf80.png">

この論文はhierarchical multi-scale attention approachを提案して、マルチスケール入力及びAttention Mechanismを使用することで、2020年のSOTAモデルになりました。
先行研究と比べると、提案手法は計算量及びトレーニング時間を減らすことができました。













