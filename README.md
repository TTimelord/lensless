# lensless

## Real Experiment
The folder "flatcam" contains code and data for testing lensless cameras with amplitude masks proposed in FlatCam paper.

You should first calibrate the lensless camera with stripped patterns as described in the FlatCam paper:

1. Generate the stripped patterns by running "picture_get_cv.py". You can modify the variable N in that script to adjust the size of the picture you want to reconstruct.
2. Run "rotation_calibration.py" to calibrate the angle of rotation.
3. Run "phi_get_main.py" to calibrate the matrix. You should adjust the parameters like the angle of rotation.

After the calibration, "calib.mat" will store the calibrated parameters. 

Then you can run "display_camera.py" to display a picture on the screen and the measurement of the flatcam will be stored. 

Finally you can use "demo.py" to reconstruct the picture from the measurement using the parameters you have calibrated beforehand.

## Simulation
The folder "flatcam_simulation" contains code and data for a simple simulation of the flatcam. The main idea is to use convolution to simulate the imaging process. **cupy** is used to accelerate the simulation process. The functions of the scripts in that folder are described in their file names.
