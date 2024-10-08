# WLaSDI
Recent work in data-driven modeling has demonstrated that a weak formulation of model equations enhances the noise robustness of a wide range of computational methods. In this paper, we demonstrate the power of the weak form to enhance the LaSDI (Latent Space Dynamics Identification) algorithm, a recently developed data-driven reduced order modeling technique.

We introduce a weak form-based version WLaSDI (Weak-form Latent Space Dynamics Identification). WLaSDI first compresses data, then projects onto the test functions and learns the local latent space models. Notably, WLaSDI demonstrates significantly enhanced robustness to noise. With WLaSDI, the local latent space is obtained using weak-form equation learning techniques. Compared to the standard sparse identification of nonlinear dynamics (SINDy) used in LaSDI, the variance reduction of the weak form guarantees a robust and precise latent space recovery, hence allowing for a fast, robust, and accurate simulation. We demonstrate the efficacy of WLaSDI vs. LaSDI on several common benchmark examples including viscid and inviscid Burgers’, radial advection, and heat conduction. For instance, in the case of 1D inviscid Burgers’ simulations with the addition of up to 100% Gaussian white noise, the relative error remains consistently below 6% for WLaSDI, while it can exceed 10,000% for LaSDI. Similarly, for radial advection simulations, the relative errors stay below 16% for WLaSDI, in stark contrast to the potential errors of up to 10,000% with LaSDI. Moreover, speedups of several orders of magnitude can be obtained with WLaSDI. For example applying WLaSDI to 1D Burgers’ yields a 140X speedup compared to the corresponding full order model.

## Citation
[Tran, April, Xiaolong He, Daniel A. Messenger, Youngsoo Choi, and David M. Bortz. "Weak-form latent space dynamics identification." Computer Methods in Applied Mechanics and Engineering 427 (2024): 116998.](https://doi.org/10.1016/j.cma.2024.116998)

## Dependencies

These packages are verified for compatibility:
* Python: 3.10.9
* TensorFlow: 2.13.0
* Numpy: 1.24.3
* Scipy: 1.9.3
* Sklearn: 0.23.2
* Pandas: 1.5.2
* Matplotlib: 3.6.2
* Seaborn: 0.12.2
* Pickle: 0.7.5
* Pytorch: 2.2.2
* Sympy: 1.11.1
* MFEM

## Instructions
There are four examples included:
* 1D Burgers Equation 
* 2D Burgers Equation 
* Radial Advection (MFEM example 9, problem 3)
* Time-dependent Diffusion (MFEM example 16)
  
Each folder contains instructions for generating the data, training the neural networks or applying a linear data-compression technique, and using WLaSDI. For each example, generate the training data by using the "Build" file for the 1D and 2D Burgers examples. For radial advection and diffusion, refer to the detailed instructions in the problem folders for MFEM setup. Each folder includes a Jupyter notebook that explains the linear compression case. For the neural network models, please run the corresponding train scripts found in each problem folder.

Two versions of WLaSDI are available: the "WLaSDI.py" class with the WSINDy implementation (from MathBioCU/WSINDy_ODE) and the "WLaSDI_wendy.py" class with the WENDy implementation (from MathBioCU/WENDy). Both implementations work well across the four examples, with comments provided within each class. Example usage is demonstrated in the Jupyter notebooks included in each example folder.

If you have any questions or comments, please contact April Tran at chi.tran@colorado.edu

## For LC Lassen users
[OpenCE 1.8.0](https://lc.llnl.gov/confluence/pages/viewpage.action?pageId=747700633) has been verified for compatibility. An example bash script is provided for the 1D Burgers case, and other cases are configured similarly. For additional help, refer to the documentation or contact support.

## Acknowledgement
This work was supported in part by a Rudy Horne Fellowship to AT. This work also received partial support from the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Computing Research, as part of the CHaRMNET Mathematical Multifaceted Integrated Capability Center (MMICC) program, under Award Number DE-SC0023164 to Y. Choi at Lawrence Livermore National Laboratory, and under Award Number DE-SC0023346 to D.M. Bortz at the University of Colorado Boulder. Lawrence Livermore National Laboratory is operated by Lawrence Livermore National Security, LLC, for the U.S. Department of Energy, National Nuclear Security Administration under Contract DE-AC52-07NA27344.

## Licence
WLaSDI is distributed under the terms of the MIT license. All new contributions must be made under the MIT. See
[LICENSE-MIT](https://github.com/MathBioCU/WLaSDI/blob/main/LICENSE-MIT)

LLNL Release Nubmer: LLNL-CODE-867254
