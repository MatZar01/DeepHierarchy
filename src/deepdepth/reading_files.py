import pickle
import numpy as np

def read_CSM(file_path_name, CSM_index=None):
    with open(file_path_name, 'rb') as f:
        matrices_list = []
        while True:
            try:
                matrix = pickle.load(f)
                if isinstance(matrix, dict):
                    np.random.seed(42)
                    random_noise = (np.random.rand(matrix['sim'].shape[0], matrix['sim'].shape[1]) - 0.5) / 10e6
                    symmetrical_noise = (random_noise + random_noise.T) / 2
                    matrix = matrix['sim'] + symmetrical_noise
                    return matrix
                matrices_list.append(matrix)
            except EOFError:
                break   
    matrices_array = np.array(matrices_list)
    if len(matrices_array) > 1:
        matrices_array = matrices_array[:-1]
        matrix = matrices_array[CSM_index]
    return matrix
        

def process_CSM(similarity_matrix, mode='NCSM'):
    if mode == 'NCSM' or mode == 'WCSM':
        similarity_matrix = (similarity_matrix - np.min(similarity_matrix)) / (np.max(similarity_matrix) - np.min(similarity_matrix))
    if mode == 'CCSM':
        similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2
    np.fill_diagonal(similarity_matrix, 1)
    return similarity_matrix