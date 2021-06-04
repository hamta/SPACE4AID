# SPACE4AI-D

A Design-time Tool for AI applications Resource Selection  in Computing Continua

### --- INSTALL WITH conda
1. Create the environment from the environment.yml file:
```conda env create -f environment.yml```
2. Activate the new environment:
```conda activate S4AI```
3. Verify that the new environment was installed correctly:
```conda env list```
4. To deactivate the environment:
```conda deactivate```

### --- EXPORT THE CURRENT conda ENVIRONMENT (across platforms)
Note: the existing environment.yml file will be overwritten
1. ```conda env export --from-history > environment.yml```

### --- REMOVE THE EXISTING conda ENVIRONMENT
1. Remove the environment:
```conda remove --name S4AI --all```
2. Verify that the environment was removed:
```conda env list```

### --- (alternative) INSTALL and RUN in a docker container
1. Build docker image
```docker build -t space4ai .```
2. Run container (interactive mode)
```docker run -it --name s4ai space4ai```

### --- GENERATE DOCUMENTATION
1. Generate html and latex documentation
```doxygen doc/Doxyfile```
2. The main page of html documentation is in doc/html/index.html
