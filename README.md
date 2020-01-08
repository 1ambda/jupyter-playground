# Jupyter Playground

## Environment

```bash
pyenv virtualenv 3.7.2 jupyter-playground
pyenv activate jupyter-playground

pip install JPype1-py3
pip install konlpy
pip install nltk

pip install matplotlib
pip install numpy
pip install sklearn
pip install gensim
pip install pandas

# https://www.tensorflow.org/install
pip install tensorflow

# https://jupyter.readthedocs.io/en/latest/install.html
pip install jupyter
pip install jupyterlab
pip install tqdm

jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter labextension install jupyter-matplotlib

jupyter lab --port=8080 --ip=127.0.0.1
```
