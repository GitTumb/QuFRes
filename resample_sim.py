from encoding import encode, depatchify,get_numQubits
from processing import circuit_builder, QuDownsample, QuDownsample_2D, QuDownsample_MD, QUpsample_MD
from postprocessing import *

# Dictionary of available circuits
circuit_library = {
    'downsample_1D': QuDownsample,
    'downsample_2D': QuDownsample_2D,
    'downsample_MD': QuDownsample_MD,
    'upsample': QUpsample_MD
}

class Resampling_Sim:
    """
    A class to perform quantum circuit simulations for resampling tasks.
    
    Attributes:
    - states (np.ndarray): Quantum states obtained from signal encoding.
    - norms (np.ndarray): Normalization factors.
    - task (str): The chosen quantum processing task.
    - circuits (list of QuantumCircuit): Generated quantum circuits for simulation.
    - logbook (dict): Stores information about the simulation (e.g., shots, frequencies).
    """

    def __init__(self, signal, task,params,patch_shape=None):
        """
        Initializes the Resampling_Sim object.

        Params:
        - signal (np.ndarray): The input signal to encode.
        - task (str): The quantum task to be performed.
        - patch_shape (tuple, optional): Shape of patches if patching is applied.
        - params (dict, optional): Parameters for the quantum circuits.

        """
        self._input_size = get_numQubits(signal)
        self._states, self._norms = encode(signal, patch_shape=patch_shape)
        self._patch_shape = patch_shape
        self._task = task
        self._isPatched = self._patch_shape is not None and self._patch_shape != self._states.shape
        self._circuits = circuit_builder(self._states, circuit_type=circuit_library[self._task], params=params, patches=self._isPatched)
        self._logbook = {'task': self._task, 'patches': self._isPatched}

    @property
    def states(self):
        """Returns the quantum states."""
        return self._states

    @property
    def norms(self):
        """Returns the normalization factors."""
        return self._norms

    @property
    def task(self):
        """Returns the processing task."""
        return self._task

    @property
    def logbook(self):
         """Returns the simulation logbook."""
         return self._logbook
        

    def run(self, shots, seed=None):
        """
        Runs the quantum circuits and updates the measurement frequencies.

        Params:
        - shots (int): Number of shots (repetitions) for the simulation.
        - seed (int, optional): Random seed for reproducibility.

        Updates:
        - logbook['frequencies']: Updates the frequencies with new measurements.
        - logbook['shots']: Increases the total number of shots.
        """
        if self._isPatched and 'shots' in self._logbook:
            new = [f*shots for f in  patch_run_circ(self._circuits, S=shots, seed=seed)]
            old = [f*self._logbook['shots'] for f in self._logbook['frequencies']]
            self._logbook['frequencies'] = [(x+y)/(self._logbook['shots'] + shots) for x,y in zip(new,old)]
            self._logbook['shots'] += shots

        elif not self._isPatched and 'shots' in self._logbook:
            new = run_circ(self._circuits, S=shots, seed=seed)
            old = self._logbook['frequencies']
            self._logbook['frequencies'] = (old * self._logbook['shots'] + new * shots) / (self._logbook['shots'] + shots)
            self._logbook['shots'] += shots

        elif self._isPatched and 'shots' not in self._logbook:
            new = patch_run_circ(self._circuits, S=shots, seed=seed)
            self._logbook['frequencies'] = new
            self._logbook['shots'] = shots

        else:
            new = run_circ(self._circuits, S=shots, seed=seed)
            self._logbook['frequencies'] = new
            self._logbook['shots'] = shots

        print('Simulation completed.')
        return True

    def reconstruct(self):
        """
        Reconstructs the signal from the stored measurement frequencies.

        Returns:
        - np.ndarray: The reconstructed signal.

        Raises:
        - Exception: If no measurement frequencies are available.
        """
        if 'frequencies' not in self._logbook:
            raise Exception('No signal to reconstruct.')

        if self._isPatched:
            d = len(self._logbook['frequencies'][0].shape)
            out_sig = patch_vec2sig(d=d, patch_freqs = self._logbook['frequencies'],norms=self._norms)
            shape = tuple(out_sig.shape[0]*x for x in out_sig.shape[1:])

        else:
            d = len(self._logbook['frequencies'].shape)
            out_sig = vec2sig(d=d, freqs=self._logbook['frequencies'], norm=self._norms)
            shape = out_sig.shape
            
        
        #Eprint(get_numQubits(out_sig)-self._input_size)
        self._sig = depatchify(out_sig,out_shape=shape)
        self._output_size =  get_numQubits(self._sig)
        norm_factor = 2**(self._output_size-self._input_size)  #renormalize by factor 2**(nTilde)
        self._sig = self._sig*norm_factor
        print('Signal successfully reconstructed!')
        return True



    @property
    def output(self):
        """Returns the reconstructed signal if available."""
        if not hasattr(self, "_sig"):
            raise AttributeError("Signal has not been reconstructed yet.")
        return self._sig