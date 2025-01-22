import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import transpile, Aer
from qiskit.circuit.library import QFT
from encoding import StateVector,get_numQubits




Circuit = QuantumCircuit #Usefull alias






#=#=#= Circuits #=#=#=

def MD_QFT(size:int, circuit,nSub:int ,d:int, inverse:bool = False)->None:
    
    '''
    Implements a multidimensional Quantum Fourier Transform (MD-QFT) on a multipartite quantum register.
    Params:
    - size (int): Number of qubits per QFT subregister.
    - circuit (QuantumCircuit): Quantum circuit where the QFT will be applied.
    - nSub (int): Number of qubits per subregister.
    - d (int): Number of subregisters.
    - inverse (bool): Whether to apply the inverse QFT.
    Returns:
    - None: The MD-QFT is applied directly to the circuit.
    '''

    label = 'iQFT' if inverse  else 'QFT' #Either QFT or Inverse

    std_QFT = QFT(num_qubits=size, approximation_degree=0, do_swaps=True,
               inverse=inverse, insert_barriers=True, name = label) # Standard, i.e.: one-dimensional, QFT

    for i in range(d):
        circuit.append(std_QFT,qargs= range(i*nSub,(i*nSub+size)))

    return None
    



def QuDownsample(state: StateVector,nTilde:int,Hadamard:bool = True)->Circuit:

    '''
    Builds a quantum circuit for the downsampling of a 1D signal.
    Params:
    - state (StateVector): Quantum state encoding the input signal.
    - nTilde (int): Number of qubits to discard.
    - Hadamard (bool): Whether to apply Hadamard gates. Default is True.
    Returns:
    - qc (QuantumCircuit): Quantum circuit performing the downsampling.
    '''

    nEnc = get_numQubits(state) #encoding qubits
    nDown = nEnc - nTilde   #output qubits
    q = QuantumRegister(nEnc)
    c = ClassicalRegister(nDown)
    qc = QuantumCircuit(q,c)
    
    qc.initialize(state)
    if Hadamard:
        qc.h(q)
        qc.barrier()
    
    QFT_Enc = QFT(num_qubits=nEnc, approximation_degree=0, do_swaps=True,
               inverse=False, insert_barriers=True,name = 'QFT')
    
    QFT__Down_dagg = QFT(num_qubits = nDown, approximation_degree = 0, do_swaps = True,
                  inverse = True, insert_barriers = True, name = 'QFT-inverse')
    
    qc.append(QFT_Enc,qargs=q)
    qc.append(QFT__Down_dagg,qargs=q[0:nDown])

    qc.barrier()
    if Hadamard:
        qc.h(q[0:nDown])
    qc.measure(q[0:nDown],c)
    
    return qc


def QuDownsample_2D(state,nTilde:int,Hadamard:bool = True)->Circuit:
   
    '''
    Builds a quantum circuit for the downsampling of a 2D signal.
    Params:
    - state (StateVector): Quantum state encoding the input signal.
    - nTilde (int): Number of qubits to discard per subregister.
    - Hadamard (bool): Whether to apply Hadamard gates.
    Returns:
    - qc (QuantumCircuit): Quantum circuit performing the 2D downsampling.  

    '''
    nEnc = get_numQubits(state)
    nDown = nEnc - 2 *nTilde
    q = QuantumRegister(nEnc)
    c = ClassicalRegister(nDown)
    qc = QuantumCircuit(q,c)
    
    qc.initialize(state)
    if Hadamard:
        qc.h(q)
        qc.barrier()
    
    QFT_Enc= QFT(num_qubits=nEnc//2, approximation_degree=0, do_swaps=True,
               inverse=False, insert_barriers=True,name = 'QFT')
    
    QFT_dagg_Down = QFT(num_qubits = nDown//2, approximation_degree = 0, do_swaps = True,
                  inverse = True, insert_barriers = True, name = 'iQFT')
    
    qc.append(QFT_Enc,qargs= range(0,nEnc//2))
    qc.append(QFT_Enc,qargs= range(nEnc//2,nEnc)) #MD-QFT on encoding register 
    
    qc.append(QFT_dagg_Down, qargs = range (0,nDown//2))
    qc.append(QFT_dagg_Down,qargs = range((nDown//2)+nTilde,nEnc-nTilde)) #Inverse MD-QFT on downsampled register
    
    qc.barrier()
    if Hadamard:
        qc.h(q[0:nDown//2])
        qc.h(q[(nDown//2)+nTilde:nEnc-nTilde])
    qc.measure(q[0:nDown//2],c[0:nDown//2])
    qc.measure(q[(nDown//2)+nTilde:nEnc-nTilde],c[(nDown//2):nDown])
    
    return qc



def QuDownsample_MD(state,d:int,nTilde:int,Hadamard:bool=True)->Circuit:

    '''
    Builds a quantum circuit for the downsampling of a generic d-dimensional signal.
    Params:
    - state (StateVector): Quantum state encoding the input signal.
    - d (int): Number of subregisters (signal dimensions).
    - nTilde (int): Number of qubits to discard per subregister.
    - Hadamard (bool): Whether to apply Hadamard gates.
    Returns:
    - qc (QuantumCircuit): Quantum circuit performing the downsampling.
    '''
    nEnc = get_numQubits(state)
    nDown = nEnc -  d*nTilde #Size of downsampled register
    n0 = nEnc//d  #num qubits for each subregister
    
    q = QuantumRegister(nEnc)
    c = ClassicalRegister(nDown)
    qc = QuantumCircuit(q,c)    
    qc.initialize(state)

    if Hadamard:
        qc.h(q)
        qc.barrier()
        
    MD_QFT(size = nEnc//d, circuit=qc, nSub=n0, d=d, inverse = False)  #MD-QFT Forward on the full encoding register 
    qc.barrier()
    MD_QFT(size = nDown//d, circuit=qc,nSub=n0, d=d, inverse = True) #Inverse MD-QFT on the un-discarded qubits 
    qc.barrier()


    if Hadamard:
        for i in range(d):
            qc.h(q[i*n0:i*n0+(nDown//d)])
        
    for i in range(d):
        qc.measure(q[i*n0:i*n0+(nDown//d)],c[i*nDown//d:(i+1)*nDown//d]) #Effectively discard the ntilde most significant qubits for each of the subregisters

    return qc




def QUpsample_MD(state,d:int,nTilde:int)->Circuit:

    '''
    Builds a quantum circuit for the upsampling of a d-dimensional signal.
    Params:
    - state (StateVector): Quantum state encoding the input signal.
    - d (int): Number of subregisters.
    - nTilde (int): Number of qubits for padding.
    Returns:
    - qc (QuantumCircuit): Quantum circuit performing the upsampling.
    '''
     
    nEnc = get_numQubits(state)
    nUp = nEnc +  d*nTilde
    n0 = nEnc//d
    n1 = nUp//d
  

    #instantiating the circuit
    q = QuantumRegister(nUp)
    c = ClassicalRegister(nUp)
    qc = QuantumCircuit(q,c)

    

    qc.initialize(state,qubits=q[0:nEnc])

    qc.h(q)

    MD_QFT(size = nEnc//d, circuit=qc,nSub=n0, d=d,inverse = False)   

    qc.barrier()
    for i in range((d-1)*nTilde):
        padd_Idx = i//nTilde +1   #Padding Index: determines which subregister the currently selected qubit has to pad
        for j in range(n0*(d-padd_Idx)):
            qc.swap(q[nEnc+i-j],q[nEnc+i-j-1])

    qc.barrier()

    MD_QFT(size = nUp//d, circuit=qc,nSub=n1, d=d, inverse = True)   

    qc.h(q)
    qc.measure(q,c)

    return qc 
            




def multipatch_circs(patches,circ_builder,params)->list:
    '''
    Allows for patch parallel processing, i.e.: building a distinct circuit for each of the patches in which we 
    have divided the original signal. Takes as input an iterable of patches, a circuit builder function (specifying the type of
    protocol we wish to perform) and the related parameters.

    Params:
    - patches (iterable): iterable collection of patches, i.e: partitions of the input signal;
    - circ_builder (func obj): function building the required algorithm circuit;
    - params (iterable): collection of parameters to be passed to circ_builder (beyond the patches themselves).

    Returns:
    - patch_circ (list): list of quantum circuits, one per patch.
    
    '''
    N = patches.shape[0] #Num of pathces
    patch_circ = [circ_builder(patches[i], *params) for i in range(0,N) ] #Generate a circuit for each of the patches
    return patch_circ




def circuit_builder(states, circuit_type,params, patches = False):
    '''
    Builds either a list of circuit (if more than one input state is passed) processing distinc patches or a single circuit processing
    the whole signal. 

    Params:
    - states (list of array-likes or array-like): either a list of states (rapresenting distinct patches of a single signal), or
    single array representing the encoding of the whole signal itself.
    - circuit_type (func obj): function, specifying the kind of algorithm to be performed, to be passed to multipatch_circs subroutine
    - params (iterable): iterable, preferably list of tuple with the additional parameters to be passed to the circuit_type func (MUST BE ORDERED
    ACCORDING TO THE SPECIFIC FUNCTION)
    '''
    
    if patches:
        #print("params:", params)
        #print("circuit_type:", circuit_type)
        qcs = multipatch_circs(states,circuit_type,params)
    else:
        qcs = circuit_type(states,*params)

    return qcs





