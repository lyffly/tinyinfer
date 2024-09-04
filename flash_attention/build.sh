# rm -rf build
mkdir -p build
cd build
cmake .. -G Ninja
cmake --build . --target all

# cp *.so /usr/local/lib/
