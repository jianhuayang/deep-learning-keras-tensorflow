to create environment for windows:

1 copy/paste exsiting file 'deep-learning-osx.yml' to create a new file and name it 'deep-learning-win.yml'
2 change environment name to 'deep'
3 run the following command in gitbash
```
conda create -f deep-learning-win.yml
```

ok, above didn't work


now try to create a new conda environment just for the sake of deep learning

see here for example https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/

```
conda -V
conda update conda
conda create --name deep
conda activate deep
conda install jupyter tensorflow keras 
conda install -c conda-forge jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
conda install -c conda-forge jupyter_nbextensions_configurator
jupyter nbextensions_configurator enable --user

# and then can run 
jupyter notebook
# visit http://localhost:8889/nbextensions to configue nbextensions
conda deactivate
```