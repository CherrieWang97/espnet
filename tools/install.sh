apt-get install virtualenv
apt-get install bc
make KALDI=/home/kaldi PYTHON=/opt/conda/bin/python CUDA_VERSION=9.0
make moses.done
make mwerSegmenter.done
make sentencepiece.done
