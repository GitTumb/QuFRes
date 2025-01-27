import numpy as np
from typing import Union




StateVector = Union[np.ndarray, list[float], list[complex]]


def patchify(signal, patch_shape=None):

    """
    Divides a multi-dimensional signal into smaller patches.
    Params:
    - signal (np.ndarray): The input multi-dimensional array.
    - patch_shape (tuple or None): Shape of each patch. If None, the entire signal is treated as a single patch.
    Returns:
    - patches (np.ndarray): Array of patches with shape (N, *patch_shape) where N is the number of patches.
    
    """

    signal_shape = np.array(signal.shape) #Original signal shape (S_0,S_1,S_2_...)
    d = len(signal_shape)
    if patch_shape is None or np.all(signal_shape == np.array(patch_shape)): #Single patch case
        patch_shape = signal_shape 
        patches =  signal
    else:
        patch_shape = np.array(patch_shape) 
        if np.any(signal_shape % patch_shape != 0): #Patch shapes should be divisors of orig. array shape
            raise ValueError("Signal shape is not divisible by patch shape: impossible to create patches") 
        
        num_patch_ax= signal_shape // patch_shape #Number of patches per axis
        
        temp_shapes = zip(num_patch_ax,patch_shape) 
        temp_shapes = np.array(list(temp_shapes)).flatten() #(P_0,S_0,P_1,S_1,...)

        og_order = np.arange(0,2*d) #(0,1,2,3,...)
        new_order = np.array(list(zip(np.arange(0,d),np.arange(d,2*d)))).flatten() #(0,3,1,4,2,5)
        patches = np.moveaxis(signal.reshape(temp_shapes),og_order,new_order) #(P_0,P_1,...,S_0,S_1,...)
        print(patches.shape)
        patches = patches.reshape(-1,*patch_shape)
    
    return patches


def depatchify(patches,out_shape):
     
     '''
    Reconstructs a signal from its patches by merging them back into the original shape.
    Params:
    - patches (np.ndarray): Array containing the signal patches.
    - out_shape (tuple): Desired shape of the reconstructed signal.
    Returns:
    - depatched_array (np.ndarray): Reconstructed signal of shape out_shape.

     '''

     if len(patches.shape)!=len(out_shape):
         patch_shape = np.array(patches.shape)[1:]
     else:
         patch_shape = np.array(patches.shape)

     signal_shape = np.array(out_shape)

     
     if np.prod(out_shape) != np.prod(patches.shape):
         raise ValueError('Output shape does not match patch shape')
     
     
     if out_shape is None or np.all(out_shape == patch_shape): #Single patch (i.e. entire signal) case
         depatched_array = patches
         
         
     else: 
        num_patch_ax = signal_shape//patch_shape
        d = len(out_shape)
        depatched_array = patches.reshape(*num_patch_ax,*patch_shape)
        new_order = np.arange(0,2*d) #(0,1,2,3,...)
        og_order = np.array(list(zip(np.arange(0,d),np.arange(d,2*d)))).flatten() #(0,3,1,4,2,5)
        depatched_array = np.moveaxis(depatched_array,new_order,og_order).reshape(*out_shape)
     return depatched_array



def get_numQubits(state:StateVector)->int:
    
    '''
   Computes the number of qubits required for a given state vector.
    Params:
    - state (StateVector): The quantum state vector.
    Returns:
    - n (int): Number of qubits needed.
    '''

    n = int(np.log2(len(state)))
    return n


    
def encode(signal, patch_shape=None):
    
    """
    Encodes a multi-dimensional signal as the amplitudes of a quantum state.
    Params:
    - signal (array-like): Input multi-dimensional signal.
    - patch_shape (tuple or None): Shape of each patch, or entire signal if None.
    Returns:
    - states (np.ndarray): Array containing the normalized quantum states.
    - norms (np.ndarray): Array of normalization factors.
    """

    patches = patchify(signal,patch_shape)
    
    patching_flag =  (patch_shape is not None) or np.all(signal.shape == np.array(patch_shape)) #1 if array is patched
    d = len(patches.shape) - 1 * patching_flag
    axes = tuple(np.arange(1,d+1).flatten())

    if len(axes)==len(patches.shape):
        norms = np.sum(patches)
        with np.errstate(divide='ignore', invalid='ignore'):
            tmp = patches/norms
            tmp[np.isinf(tmp)] = 0  # norm 0 implies all entries are 0, so it's 0/0

    
    else:
        #norms = np.sum(patches,axis=axes).reshape(patches.shape[0],*(2*(1,)))
        norms = np.sum(patches,axis=axes)
        with np.errstate(divide='ignore', invalid='ignore'):
            tmp = patches/norms[:,*(d*(None,))]
            tmp[np.isinf(tmp)] = 0  # norm 0 implies all entries are 0, so it's 0/0
    
    

    states = np.sqrt(tmp)
    


    return states, norms



