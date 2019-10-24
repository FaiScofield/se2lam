# !/usr/bin/bash

echo "Building DBoW2 ..."
cd Thirdparty/DBoW2
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j10

echo "Building g2o ..."
cd ../../g2o-20160424
mkdir build
mkdir install
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="../install"
make -j10
make install

echo "Done."
