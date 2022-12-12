# Reserch：スペースデブリ除去のための画像生成AIの誤差逆伝播による相対姿勢推定
球面座標系におけるカメラ座標(r,Θ,φ)を入力に、スペースデブリの画像を生成するネットワーク。
このネットワークを用いて生成された偽物のデブリ画像が、カメラで撮影された本物のデブリ画像と完全に一致するまで、誤差逆伝播によって入力座標を最適化することで相対姿勢を求める。
入力座標最適化のための修正量は、TensorFlowのGradientTapw(勾配テープ)で算出した勾配をAdamで最適化したものを用いる

そのうち書く
