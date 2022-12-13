# Reserch：スペースデブリ除去のための画像生成AIの誤差逆伝播による相対姿勢推定
球面座標系におけるカメラ座標(r,Θ,φ)を入力に、スペースデブリの画像を生成するネットワーク。

このネットワークを用いて生成された偽物のデブリ画像がカメラで撮影された本物のデブリ画像と一致するように入力座標を最適化することで相対姿勢を求める。  
入力座標最適化のための修正量は、TensorFlowのGradientTapw(勾配テープ)と誤差逆伝播によって算出した勾配をAdamで最適化したものを用いる

## 学習データ(Blender)
画像生成ネットワークの教師データは、3DモデリングソフトBlenderを用いて作成。Blenderは完全無料で誰でも利用することが出来る。  
![image](https://raw.github.com/wiki/neo-miyake/Reserch/Readme_img/9955,2.2868717863891215,0.9109514714692605.jpg)
![image](https://raw.github.com/wiki/neo-miyake/Reserch/Readme_img/clementine_thumb.png)  
あのNASAも積極的に利用しており、人工衛星やロケット月面モジュールなどの3Dモデルが[NASA 3D Resources](https://nasa3d.arc.nasa.gov/models "nasa3d")で公開されている(一部は外部ユーザーの投稿によるモデルもあるので注意)。
