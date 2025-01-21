# Standard Plane Localization using Denoising Diffusion Model with Multi-scale Guidance

This repository is the implementation of the diffusion-based approach for standard plane localization in 3D ultrasound.


## Usage  
### Dataset
The data folders should be placed as following:  
```
-- datasets
   ---organ1 
      ---case 1  
         ---volume.nii.gz   # the ultrasound volume  
         ---tangent.npz  # include the information for the GT plane parameters
      ---case 2  
      ...

-- datasplit
  ---organ1 
     --- train.txt # case name list for training
     --- val.txt # case name list for validation
     --- test.txt # case name list for testing
```

### Training
Set your own dataset information in './configure/Prototyping.yaml', and run the following commands:
```bash
python main.py
```
