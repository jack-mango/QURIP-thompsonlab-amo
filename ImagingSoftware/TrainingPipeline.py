class TrainingPipeline():

    # Things this class should do:
    #   1. Find tweezer positions
    #   2. Make crops and labels for given directory
    #   3. Train the neural network and evaluate fidelity
    #   4. Give breakdown of testing and training statistics
    #   

    def __init__(self, path, n_tweezers, n_loops):
        self.path = path
