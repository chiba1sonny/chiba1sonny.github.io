---
layout: post
title: 画像セグメンテーションに向けたMeanshift(ミーンシフト)
image: /img/ms.jpg
tags: ["image segmentation", "machine learning"]
---

## 0.Meanshiftによるclustering
先に、Sklearn.cluster.MeanShiftを利用して、あるデータをクラスタリングしてみます。データ生成するにはSklearnのdatasets.make_blobsを使用しました。生成されるサンプル数を2000にして、
クラス数を４、サンプルごとの特徴数を２、cluster_stdをそれぞれ0.5, 1.0, 1.5, 2.0にしました。

こちらは生成されたデータです

![image](https://user-images.githubusercontent.com/68838083/197840354-a557dab6-e742-4693-a230-7b2f3ac42e74.png)


```py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
from sklearn.cluster import estimate_bandwidth
from sklearn import datasets

#データを作る
X,y = datasets.make_blobs(n_samples=2000, centers=4, n_features=2,
                           cluster_std = [0.5, 1.0, 1.5, 2.0])
plt.scatter(X[:,0],X[:,1],s = 10,alpha = 0.8,c='k')
plt.grid()

#bandwidthを計算する
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=5000)
print(bandwidth)

#Meanshiftをフィットする
ms = MeanShift(bandwidth = bandwidth, bin_seeding = True)
ms.fit(X)
labels = ms.fit_predict(X)

#結果
plt.figure()
plt.scatter(X[:,0], X[:,1], c = labels)
plt.axis('equal')
plt.title('meanshift prediction')
plt.show()

print(bandwidth)
```
結果

![image](https://user-images.githubusercontent.com/68838083/197840549-f5ed882c-19f4-4cf4-9418-851fa2824120.png)

bandwidth は 2.9507367943444662

## 1.2012年前のイメージセグメンテーション
- 2012年、ImageNetコンテストILSVRC（2010年より開催）では、1000クラス識別タスクで、深層学習を用いた方法が圧勝しました。その後も、急激な性能向上が続き、色なコンピュータービジョンタスクの状況を変えてきました。現在の深層学習におけるセマンティックセグメンテーションといえば、画像中の各ピクセルがどの物体に属するのかを推定するタスクと言えます。インスタンスセグメンテーションとは、画像中のすべてのオブジェクトに対して、クラスラベルを予測し、同じクラスのオブジェクトであっても一意のIDをつけます。パノプティックセグメンテーションとは、上記の2つを組み合わせタスクです。

![Screenshot from 2022-10-24 18-12-34](https://user-images.githubusercontent.com/68838083/197491433-960ff7f7-0181-472c-961b-ce926fb413ee.png) +
![Screenshot from 2022-10-24 18-13-14](https://user-images.githubusercontent.com/68838083/197491552-12116528-06cc-4d52-a940-73bc8b3dd823.png) ーーー
![Screenshot from 2022-10-24 18-13-49](https://user-images.githubusercontent.com/68838083/197491719-17849f7b-373d-4527-84c2-b29f06a4e201.png)

- 深層学習学習の発展によって、イメージセグメンテーションも進歩しています。しかし、2012年前のイメージセグメンテーションは別の話であります。その時のイメージセグメンテーションの目的は「Separate image into conherent "objects"」, 「Identify similar image regions」と言えます。その時、主にclustering（clustering is the problem of grouping similar data points and represent them with a single token）で、イメージセグメンテーションを行うことでした。

![Screenshot from 2022-10-24 18-20-23](https://user-images.githubusercontent.com/68838083/197493040-74a9e19e-c41c-45a0-8321-4be1b585af33.png)ーーー
![Screenshot from 2022-10-24 18-20-53](https://user-images.githubusercontent.com/68838083/197493110-41d5241f-27f4-44c3-aa18-34d8b53d95ad.png)

## 2.Meanshiftの原理について
Meanshiftは、カーネル密度推定（kernel density estimation）を用いたデータ解析手法です。イメージセグメンテーション、画像平滑化などに応用されていました。
#### Meanshiftの流れ
| 記法 | 意味 |
| --- | --- |
| $R^d$ | ｄ次元空間 |
| $x_i$ | サンプル点 |
| $i$ | $i=1,2,...n$ |
| $M_h(x)$ | サンプル点$x$に対するMeanshiftベクター |
| $k$ | サンプル点$x$が含まれる円状区域の点数 |
| $h$ | サンプル点$x$が含まれる円状区域の半径 |
| $S_h(x)$ | サンプル点$x$が含まれる円状区域 |

$R^d$内に$n$個のサンプル点 $x_i$ があり、その中の1つの点 $x$ に対するMeanshiftベクターの基本形式は以下のようにである

$$M_h(x)=1/k \sum_{x_i∈S_h(x)}^k (x_i-x) $$

こちらの $S_h(x)$ は半径は $h$ である円状区域で、円の中心点は $x$ であり、中に $k$ 個のサンプル点が含まれています。この円の中心点とすべての $k$ 個の点のベクター和はMeanshiftになります。下の図で黄色の矢印でMeanshiftを示します。青い円状区域は $S_h(x)$ 、中心点は $x$ です。

![Screenshot from 2022-10-25 16-41-19](https://user-images.githubusercontent.com/68838083/197713179-1c6dc638-4280-4c35-a62c-2593824646e2.png)

Meanshiftの流れは：
- 現在の $S_h(x)$ で中心点は $x$ のベクター和を計算する（$S_h(x)$ の重心を計算する）　
- 中心点は $x$ はその重心まで移動する
- 移動量が十分に小さくなれば、最初に戻る

Meanshiftはiterativeです。特定のサンプル点のある区域の重心を計算し、それに移動します。移動量が十分に小さいくなれば最初に戻ります。下の図では、青い円状区域は計算される区域で、windowとも呼ばれます。

![Screenshot from 2022-10-25 16-57-45](https://user-images.githubusercontent.com/68838083/197716838-2cc0f9d9-5b87-4a21-acff-a8b0057e33d2.png)

そして、反復プロせうの終了条件を満たす場合、すべてのwindowがある点で集合すれば、一集合になり、clusteringされます。以下の図のようです。

![Screenshot from 2022-10-25 17-05-48](https://user-images.githubusercontent.com/68838083/197718535-d04fba09-6f97-4080-a110-2f97f85fef42.png)

下の図でmeanshiftアルゴリズムによるclusteringを表現できます。

![Screenshot from 2022-10-25 17-07-41](https://user-images.githubusercontent.com/68838083/197718934-b4353523-6afc-4a5a-a4db-22164d1da3c5.png)

## 3.Meanshiftによるイメージセグメンテーションの実装
Meanshiftアルゴリズムによるイメージセグメンテーションを体験してみました。Sklearnにじっそうされているmeanshiftとbandwidthを計算できるestimate_bandwidthを利用しました。注意すべきことはestimate_bandwidthのn_samplesです。n_samplesとは、ランダムにn個のサンプルを選択してbandwidthを計算するハイパーパラメータのことで、大きくすれば計算量が非常に大きくなります。ディフォルトはNone, すべてのデータで計算することです。

元画像

![image](https://user-images.githubusercontent.com/68838083/197722667-990853f9-c911-4f83-b91f-422a9e2da0fe.png)

```py
import matplotlib.pyplot as plt
image = plt.imread('icon.jpg')
plt.figure(dpi=150)
plt.title('original image')
plt.imshow(image)

from sklearn.cluster import MeanShift
from sklearn.cluster import estimate_bandwidth
import numpy as np
from pylab import *

bandwidth1 = estimate_bandwidth(image, quantile = 0.2, 
                                n_samples = 1000 )

meanshift = MeanShift(bandwidth = bandwidth1, bin_seeding = True, 
                      n_jobs = -1, cluster_all = True)

meanshift.fit(img)

label = meanshift.labels_

label = label.reshape(720,1280)

imshow(label)
```
結果

![image](https://user-images.githubusercontent.com/68838083/197722581-5cf963d9-c20b-458b-ab01-f2cae992e466.png)

skimageのastronaut画像にmeanshiftを試しました。上の結果は、n_samplesを1000にしたので、estimate_bandwidthは数分間で終わりました。今回、n_samplesをディフォルト、すなわち、すべてのデータを計算に使いました。計算時間は大体３時間でした。

元画像

![astronaut](https://user-images.githubusercontent.com/68838083/197723666-c83c92fb-2087-47e6-9144-816080343a1a.png)

結果

![download](https://user-images.githubusercontent.com/68838083/197723742-704b9647-c233-4cdb-88bd-8c7daf2fc561.png)

#### References
1. [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.estimate_bandwidth.html#sklearn.cluster.estimate_bandwidth)
2. [図など](http://vision.stanford.edu/teaching/cs131_fall1617/lectures/lecture13_kmeans_mean_shift_cs131_2016)
3. [図など](https://www.ipb.uni-bonn.de/html/teaching/photo12-2021/2021-pho1-12-segmentation.pptx.pdf)
