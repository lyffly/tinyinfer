# rm -rf build
# mkdir -p build && cd build
# cmake ..
# make -j8
# make install

python3 setup.py bdist_wheel
pip install dist/*.whl
