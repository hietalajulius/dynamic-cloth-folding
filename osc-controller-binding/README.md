## Operational Space Controller (OSC) + python binding

These files are for calculating the OSC (http://khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf) controller torques for the robot. The `include/osc_step.h` logic can and should be used in the real-world implementation of the controller (ros-control etc.). The `src/main.cpp` is for converting the numpy values obtained from simulation into eigen vectors used by the controller. The controller should work for any 7DOF robot but the experiments of the paper have been performed with a Franka Emika Panda.

### Dependencies

`sudo apt-get install -y ninja-build libeigen3-dev`

You may need to `sudo ln -s /usr/include/eigen3/Eigen /usr/include/Eigen` if Eigen does not seem to be found (https://github.com/opencv/opencv/issues/14868#issuecomment-520142022)