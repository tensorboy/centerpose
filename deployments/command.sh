rm build -r 
mkdir build 
cd build 
cmake ..
make -j12

cd .. 
./buildEngine -i model/ckpt1.onnx -o model/ckpt1.engine

./runDet -i model/ckpt1.engine --img /home/tensorboy/Documents/centerpose/images/33823288584_1d21cf0a26_k.jpg

