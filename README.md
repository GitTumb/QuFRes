# QuFRes
This repository contains the [Qiskit](https://github.com/Qiskit) simulation code for the quantum frequency resampling toolkit. Specifically, it provides 4 distinct Python modules, and an example Jupyter notebook. The modules are organized as follows:
1. `encoding` provides the basic functions for signal-to-quantum state conversion and data patching utilities
2. `processing` defines the actual Qiskit circuital implementations of the resampling protocols (we distinguish between 1D downsampling, 2D downsampling, MD downsampling, with M arbitrary dimension and generic upsampling) ;
3. `postprocessing` contains the routines related to circuit execution and output retrieval
4. `resample_sim` defines a high-level simulation class that integrates all components, automating the resampling process.

Finally, the demo notebook demonstrates the basic functionality of the package, showing a simple one-dimensional example.
