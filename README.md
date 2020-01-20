# Jupyter Playground

## Notebooks

- [공공 데이터 - 아파트 매매 실거래자료 수집](https://github.com/1ambda/jupyter-playground/blob/master/exploration-pubilc-data-gov/crawl-apt-trade.ipynb)


## Environment

```bash
pyenv virtualenv 3.7.2 jupyter-playground
pyenv activate jupyter-playground

pip install JPype1-py3
pip install konlpy
pip install nltk

pip install matplotlib
pip install seaborn

pip install numpy
pip install sklearn
pip install gensim
pip install pandas

pip install beautifulsoup4

# https://www.tensorflow.org/install
pip install tensorflow # 2.1

# https://pytorch.org/get-started/locally/
pip install torch torchvision

# https://jupyter.readthedocs.io/en/latest/install.html
pip install jupyter
pip install jupyterlab
pip install tqdm

jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter labextension install jupyter-matplotlib

jupyter lab --port=8080 --ip=127.0.0.1
```
