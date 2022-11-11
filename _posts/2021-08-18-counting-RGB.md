---
layout: post
title: Our First Post!
image: /img/dataset.png
tags: ["machine translation","natural language processing","natural language generation"]
---

[こちら](https://qiita.com/chiba1sonny/items/9a16fbb3e8136e3f983c)にアノテーションツールLabelmeを使ってデータセットを作る方法を記載しました。作ったデータセットの属性を了解することは、データ処理やモデル構築に有益です。この記事では、自作画像セグメンテーションデータセットの画像ごとにあるRGB値とそれぞれが占めるピクセル値を統計する方法と全データセットにあるRGB値を入手する方法を記載します。

### 1.画像ごとにあるRGB値とそれぞれが占めるピクセル数
使う画像はPascal VOC2012データセットのgt画像です。

![nisgel.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/1668082/bc76da96-ddc1-44ba-6e6d-1a8aed661125.png)

```py
from PIL import Image
import numpy as np

img = Image.open('pascal/nisgel.png').convert('RGB')
arr = np.array(img)
colours, counts = np.unique(arr.reshape(-1,3), axis=0, return_counts=1)
print(colours,counts)
```
このスクリプトで画像中にあるRGB値とそれぞれが占めるピクセル数を算出できます。
結果：

```
[[  0   0   0]
 [128   0   0]
 [224 224 192]] [223955  26602  12612]
```

### 2.全データセットのRGB値の統計
使う画像は以下の二つのPascal VOC2012データセットのgt画像です。

<img width="400" alt="mojikyo45_640-2.gif" src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/1668082/3c54bd9c-f525-bb04-ee66-f3bde0cfe825.png">  

<img width="400" alt="mojikyo45_640-2.gif" src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/1668082/6e3f13fe-da6a-dde6-b4a8-e4f4aa7cdea0.png">

```py
import os
from PIL import Image
import numpy as np

base = 'pascal'
all_image = os.listdir(base)
result = []
for img in all_image:
    image = Image.open(f'{base}/{img}').convert('RGB')
    arr = np.array(image)
    colors = arr.reshape(-1, 3)
    result.extend(colors)
    colors = np.unique(result, axis=0, return_counts=1)
    result = colors[0].tolist()
print(result)
```

これで、全データセットのRGB値は得られます。
結果：

```
[[0, 0, 0], [128, 0, 0], [128, 128, 0], [224, 224, 192]]
```

### 以上です。
