from collections import Counter
import numpy as np
from multiprocessing import Pool
from functools import partial
from qiskit import Aer,transpile


            
def get_freqs (res, n_outcomes):
    '''
    Converts raw measurements results into empirical occurence frequencies for each of the possible outcomes.

    Params:
    - res (iterable): array of raw measurement data;
    - n_outcomes(int): size of outcome space.

    Returns:
    - freqs(numpy array): occurence frequencies from given measurements.
    '''
    freqs = np.zeros((2**n_outcomes))
    counts = Counter(res)
    for k in counts.keys():
        freqs [k] = counts[k]/len(res)
    return freqs


def run_circ(qc,S:int,seed:int=None)->np.ndarray:

    """
    Simulates the execution of a quantum circuit and returns the frequency of measurement outcomes.

    Params:
    - qc (QuantumCircuit): The quantum circuit to be executed.
    - S (int): Number of shots, i.e., the number of repetitions of the experiment.
    - seed (int, optional): Random seed for reproducibility of the simulation results.

    Returns:
    - np.ndarray: An array containing the relative frequencies of each possible measurement outcome.
    """
    
    backend_options = {"seed_simulator": seed} if seed is not None else {} #Optional: set seed for simulations
    simulator = Aer.get_backend('aer_simulator') #Instantiate simulator
    
    qc = transpile(qc, simulator) #Transpile circuit

    
    job = simulator.run(qc, shots = S, run_options=backend_options,memory=True) #Run simulation, collect raw measurement outcomes
    outcomes = [int( m,2) for m in job.result().get_memory()] #Convert measurment outcomes (bitstrings) into integers
    
    out_freqs = get_freqs(outcomes,qc.num_clbits)
       

    return out_freqs


def patch_run_circ(qcs,S:int,seed:int=None,num_workers=4):

    partial_run_circ = partial(run_circ, S=S,seed=seed)

    with Pool(processes=num_workers) as pool:
        results = pool.map(partial_run_circ, qcs)

    return results


def vec2sig(freqs,d,norm):

    """
    Reconstructs a signal from the measurement outcomes of a quantum circuit simulation.

    Params:
    - qc (QuantumCircuit): The quantum circuit to be executed.
    - S (int): Number of shots, i.e., the number of repetitions of the experiment.
    - seed (int): Random seed for reproducibility.
    - d (int): Dimensionality of the original signal.
    - norm (np.ndarray): Normalization factors for the signal reconstruction.

    Returns:
    - np.ndarray: Reconstructed signal, reshaped according to the specified dimensionality.
    """
    
    out_values = freqs *norm
    
    out_sig = out_values.reshape(d*(int(np.power(len(freqs),1.0/d)),))

    return out_sig


def patch_vec2sig(patch_freqs, d, norm, num_workers=4):
    """
    Reconstructs a signal from the measurement results of multiple quantum circuits,
    each corresponding to a signal patch, using parallel processing.

    Params:
    - patch_freqs (list of np.ndarray): List of frequency results from quantum circuits.
    - d (int): Dimensionality of the original signal.
    - norm (np.ndarray): Normalization factors for each patch.
    - num_workers (int, optional): Number of worker processes for parallel execution.

    Returns:
    - np.ndarray: Reconstructed signal patches in parallel.
    """
    # Fixing d and norm 
    partial_vec2sig = partial(vec2sig, d=d, norm=norm)

    #Parallel execution
    with Pool(processes=num_workers) as pool:
        out_patches = pool.map(partial_vec2sig, patch_freqs) #parallelization on elements of a list

    return np.array(out_patches)

