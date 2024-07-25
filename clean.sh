rm -rf __pycache__
rm -rf build
rm -rf dist
rm -rf bin
rm -rf lib
rm -rf tinyinfer.egg-info
rm -rf tinyinfer/tinyinfer.egg-info
rm -rf tinyinfer/__pycache__
rm -rf tinyinfer/*.so
rm -rf tinyinfer/*/*.so
rm -rf data/*.bin

pip uninstall tinyinfer -y
