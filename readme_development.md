# Testing

## VScode
Testing can be configured in vscode using the .vscode/settings.json configuration file from the repo (will be automatically used when in the root directory). 

You can then go to the testing tab of vscode to discover and then run all tests. If nothing comes up, there is most likely an error in one of the files, preventing them from being discovered. Any errors can be found in the output window when selection Python Test. Setting it up in vscode can sometimes take a bit of time, but once set up it provides a really nice interface for testing where you can also use the debugger inside tests.

Pytest runs both all the tests in the Tests/ folder, as well as all example docstrings. This is configured in the pytest.ini file containing the following:
```ini
[pytest] 
addopts = --doctest-modules src/fetalbrain/
testpaths = Tests
```

However, when running or debugging individual tests in the vscode testing set-up, it also reruns all docstrings each time which makes this quite slow. For development this can be disabled by commenting out the addopts line in the pytest.ini, but should always be included before committing / pushing any changes to the repo. 


## Command line
Alternatively, you can also run pytest from the command line with:

``pytest Tests``

to also test the example docstrings in all files use:

``pytest --doctest-modules src/fetalbrain/ Tests``

This can sometimes be useful if tests are not detected by vscode to see whether the error is with pytest or with the vscode configuration. 

To also get a report of how much of the codebase is covered by tests, you can use: 

```pytest  --doctest-modules src/fetalbrain/ --cov-report term-missing --cov=src Tests```

This will provide a percentage of lines that is covered in each file and show the lines that are not yet accessed by any tests. The aim is to have this coverage percentage as high as possible. 

# Contributing
When contribution new features the following should be kept in mind
    - Minimise new dependencies, as this will break the package if something changes / is no longer supported. 
    - Make sure is mostly backward compatible, i.e. that people who have been using the code do not have to adapt their code too much when changes are introduced.

To contribute new features or bug fixes for the package use the following steps:
1. Make the changes on a new branch
   - Create a local branch and make the code changes on this branch
   - Perform local testing (see above)
     - Write additional tests for any new code
     - Ensure all tests pass
   - Push the branch to git
2. Open a pull request on Github to merge the changes onto main
   - You will have to assign an approver, this should ideally be someone else but if it can't be avoided you can also assign yourself. 
   - Before the pull request can be approved, all tests are automatically run on Github and have to all pass. The documentation is also automatically built at this stage.
   - Once the tests are passed and documentation is build, the pull request can be approved which merges the changes onto main. 

# Building Documentation
The documentation can be rebuild by going into the docs folder and from there running:
```make html```

This builds the documentation locally, which can be visualised by opening the index file located at `docs/build/html/index.html`. As the backbone for the documentation already exists, new elements can be added copying the structure of the already existing modules. 

To generate documentation for a new project, a more detailed guide is giving in `set_up_sphinx.md`, but this should not be necessary anymore for this project. 

The documentation has been largely based on the documentation of torchio, so this can be used as a reference as well. 