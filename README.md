# Reserch：スペースデブリ除去のための画像生成AIの誤差逆伝播による相対姿勢推定
球面座標系におけるカメラ座標(r,&theta;,&phi;)を入力に、その座標から見たときのスペースデブリの画像を生成するネットワーク。  

![球面座標系](https://user-images.githubusercontent.com/95911997/208359363-2af335c8-db31-4324-a839-f421bd9aae4c.jpg)

本研究ではこのカメラ座標を相対姿勢と呼ぶ。構成は以下。
2次元の入力相対姿勢から全結合層(FC)で次元を拡張し、その後畳み込み層(Conv)とアップサンプリング層(Upsampling)、スキップコネクション(Concation)を5回繰り返すことで256×256の解像度のデブリ画像を生成する。

![ネットワーク構成](https://user-images.githubusercontent.com/95911997/207896097-ec2790ea-a2c2-4175-b670-c61653525056.jpg)  

このネットワークを用いて生成された偽物のデブリ画像(**青矢印**)がカメラで撮影された本物のデブリ画像と一致するように入力座標を最適化(**橙矢印**)することで相対姿勢を求める。
入力座標最適化のための修正量は、TensorFlowの***GradientTape***(勾配テープ)と誤差逆伝播によって算出した勾配をAdamで最適化したものを用いる。  
  
![推定方法](https://user-images.githubusercontent.com/95911997/207898346-fcefde41-74a4-45af-8eda-a3bf28e79751.jpg)  
  
## 学習データ(Blender)
画像生成ネットワークの教師データは、3DモデリングソフトBlenderを用いて作成。Blenderは完全無料で誰でも利用することが出来る。
    
![](https://user-images.githubusercontent.com/95911997/207211726-6e726cb0-5e3e-40a8-9584-72b89d3c07b6.jpg)
    
NASAも積極的に利用しており、人工衛星やロケット、月面モジュールなどの3Dモデルが[NASA 3D Resources](https://nasa3d.arc.nasa.gov/models "nasa3d")で公開されている(一部は外部ユーザーの投稿によるモデル)。

## 推定時の様子  
ネットワークが推定を行う過程をgifにしてみた。出力崩れがないのを見ると生成ネットワークの精度は十分高いと言える。  
画像が動かない場合は***ページの再読み込み***または***クリックして表示***すると動く。  
  
![out](https://user-images.githubusercontent.com/95911997/209664779-e978c3e4-f1cc-45a9-ac99-5af6d6439a2a.gif)
