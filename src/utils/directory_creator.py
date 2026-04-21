import os
def create_folders(DATA_PATH, actions, n_sequences):
    for action in actions:
        for sequence in range(n_sequences):
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass

