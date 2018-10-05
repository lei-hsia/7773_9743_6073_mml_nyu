import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

notebook_filename = '../../linear_regress_m1_assign1.ipynb'
nb = nbformat.read(open(notebook_filename), as_version=4)