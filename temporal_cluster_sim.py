import cupy as cp
from mpi4py import MPI
import time
import numpy as np

# Αρχικοποίηση MPI (για επικοινωνία μεταξύ των κόμβων του Alps)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def heavy_computation(data_size):
    """Εξομοίωση του βαρύ υπολογισμού που διαρκεί 'χρόνια'"""
    # Χρήση της GPU (GH200) για πολλαπλασιασμό μεγάλων πινάκων
    a = cp.random.rand(data_size, data_size, dtype=cp.float32)
    b = cp.random.rand(data_size, data_size, dtype=cp.float32)
    # Η 'δουλειά' που γίνεται στο παρελθόν
    result = cp.matmul(a, b)
    cp.cuda.Stream.null.synchronize()
    return cp.mean(result)

# Παράμετροι εξομοίωσης
DATA_SIZE = 15000  # Μέγεθος πίνακα για GPU stress
PARADOX_TRIGGER = False # Αν γίνει True, προκαλεί κατάρρευση

if rank == 0:
    # --- PRESENT NODE ---
    print(f"[PRESENT] Initialization on Alps Cluster. Nodes available: {size-1}")
    
    # Ο Present Node 'λαμβάνει' το αποτέλεσμα πριν στείλει την εντολή
    print("[PRESENT] Waiting for 'Future-Past' result via CTC Link...")
    
    # Συγκέντρωση αποτελεσμάτων (Gather) από το 'παρελθόν'
    results = comm.gather(None, root=0)
    final_solution = np.mean(results[1:])
    
    print(f"[PRESENT] SUCCESS: Received computation result: {final_solution}")
    
    # Η κρίσιμη στιγμή της Αιτιότητας
    if PARADOX_TRIGGER:
        print("[CRITICAL] Paradox detected: Present Node refused to send signal!")
        comm.abort() # Τοπική κατάρρευση του συστήματος
    else:
        print("[PRESENT] Signal sent to the past to close the loop.")

else:
    # --- PAST NODES (Ranks 1 to N) ---
    # Προσομοίωση της λήψης του σήματος στο παρελθόν
    # Στην πραγματικότητα, οι κόμβοι ξεκινούν αμέσως
    start_time = time.time()
    
    # Εκτέλεση του υπολογισμού
    val = heavy_computation(DATA_SIZE)
    
    # Υπολογισμός 'Εντροπίας' (θερμικό φορτίο GPU)
    entropy_delta = cp.random.uniform(0.1, 0.5)
    
    print(f"[NODE {rank}] Computation complete. Entropy Delta: {entropy_delta:.4f}")
    
    # Αποστολή του αποτελέσματος 'πίσω' στο Rank 0
    comm.gather(val.get(), root=0)