# Studio-projektowe-2020
Human pose estimation using tf-pose-estimation repository
Algorithm estimating human pose and plot results after processing some dataset previously uploaded to proper folder.

## Virtual environment with installed packages
    tfpose
  
## Required packages to install:
    -cython
    -pycocotools ( cd Studio-projektowe-2020/pycocotools-2.0.0 | python setup.py install )
    -tensorflow 1.14
    -opencv 3.4.0.14
    
## Build c++ library for post processing. See : https://github.com/ildoonet/tf-pose-estimation/tree/master/tf_pose/pafprocess
    cd tf_pose/pafprocess
    swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace
    
## Dataset folder
    pictureDir = './human_dataset/'
    goodPictureDir = './goodPics/'

## Run file
    run_points.py
  
## Output
    *Preprocessed filenames in command prompt
    *result.mat
  
## Repository contains original files with instruction from its author
    https://github.com/ildoonet/tf-pose-estimation
