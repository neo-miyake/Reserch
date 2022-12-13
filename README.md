# Reserch：スペースデブリ除去のための画像生成AIの誤差逆伝播による相対姿勢推定
球面座標系におけるカメラ座標(r,Θ,φ)を入力に、スペースデブリの画像を生成するネットワーク。

このネットワークを用いて生成された偽物のデブリ画像がカメラで撮影された本物のデブリ画像と一致するように入力座標を最適化することで相対姿勢を求める。  
入力座標最適化のための修正量は、TensorFlowのGradientTapw(勾配テープ)と誤差逆伝播によって算出した勾配をAdamで最適化したものを用いる

## 学習データ(Blender)
画像生成ネットワークの教師データは、3DモデリングソフトBlenderを用いて作成。Blenderは完全無料で誰でも利用することが出来る。  
　　　　![clementine_thumb](https://user-images.githubusercontent.com/95911997/207207996-eab25d0a-1094-4670-8367-407178ab28a8.png)
![9955,2 2868717863891215,0 9109514714692605](https://user-images.githubusercontent.com/95911997/207207654-f37e3ebe-49bd-40d4-8f85-2d4f4402a03a.jpg)
![1086,0 8647583536518422,1 4192380031796643](https://user-images.githubusercontent.com/95911997/207207731-10ab7367-f8c7-4875-adfc-2c2cdb89809d.jpg)  
あのNASAも積極的に利用しており、人工衛星やロケット月面モジュールなどの3Dモデルが[NASA 3D Resources](https://nasa3d.arc.nasa.gov/models "nasa3d")で公開されている(一部は外部ユーザーの投稿によるモデルもあるので注意)。
