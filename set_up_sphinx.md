pip install sphinx (7.2.6)

Step 1: run, accepting all defaults
```
sphinx-quickstart
```

Step 2: edit conf.py (see actual conf.py file for details)
```
import sys
sys.path.insert(0, os.path.abspath('../../src'))
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.viewcode']
```


Step 3: generate rst files
```
sphinx-apidoc -o docs/src src/
```

Step 4: add the generated modules.rst in the index.rst file
```
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules
```

Step 5: generate html
```
make html
```
Step 6: view index.html in _build folder




