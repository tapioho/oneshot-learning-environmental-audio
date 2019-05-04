import numpy as np
from keras.utils import Sequence

class PairGenerator(Sequence):
    def __init__(self, X, y, batch_size=32, shuffle=True, seed=None, amount_of_pairs=1.0, neg_pair_ratio=1.0):
        """Initialize a generator
            Inputs:
                X: Training spectra
                y: Training labels
                batch_size: Number of samples per batch (use smaller batch for limited GPUs)
                shuffle: Shuffle the training data after each epoch
                seed: Use a known seed for the rng
                amount_of_pairs: Amount of pairs used from the maximum (between 0 and 1)
                neg_pair_ratio: Negative-to-positive-pair ratio (between 0 and n_classes-1)
                """
        
        # Initialize
        self.dim = [batch_size] + [k for k in X.shape[1:]]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.spectra = X
          
        # Set seed if specified
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()       
        
        self.get_pair_ids(y, amount_of_pairs=amount_of_pairs,
                             neg_pair_ratio=neg_pair_ratio)
        self.on_epoch_end()
    
    
    def get_pair_ids(self, y, amount_of_pairs, neg_pair_ratio):
        """Form training sample pair indices for generator usage"""   
        
        # Determine how many classes are used for pairs
        y = y[:,np.nonzero(np.sum(y, axis=0))[0]]  # remove unused classes
        n_classes = y.shape[1]
        n_class_samples = int(y.shape[0] / n_classes)
    
        # Calculate training set size based on the number of classes and positive pairs.
        amount_of_pairs = max(min(amount_of_pairs, 1.0), 0.0)
        neg_pair_ratio = max(min(neg_pair_ratio, n_classes - 1.0), 0.0)
        n_pairs_pos = max(int(n_class_samples/2*amount_of_pairs), 1)
        n_pairs_neg = int(n_pairs_pos * neg_pair_ratio)        
        
        # n_pairs_pos pairs for 
        N_pos = int(n_classes * (n_class_samples-n_pairs_pos) * n_pairs_pos)
        
        # Calculate neg pairs similarly but just n_pairs_neg for each 
        N_neg = int(n_classes * (n_class_samples-n_pairs_pos) * n_pairs_neg)
        N = N_pos + N_neg
        
        
        # Pre-allocate memory
        pair_ids = np.zeros((N,2), dtype=int)
        pair_labels = np.zeros(N, dtype=int)
        
        # Find indices for classes
        class_ids = np.where(y)
        
        # Form pairs
        for i in range(0, N, n_pairs_pos+n_pairs_neg):
            # Pick a random sample
            sample_id = self.rng.randint(0, class_ids[0].size)
            
            # Find samples with the same class
            same_class_ids = class_ids[1] == class_ids[1][sample_id]
            same_class_ids[sample_id] = False  # make sure our sample is not included
            
            # Make sure there is at least n_pairs_pos  pairs available for our random sample
            # NOTE: If N is too large, there will never be two positive pairs available
            #       in the last iterations and the loop gets stuck (N is calculated to avoid this)
            while np.count_nonzero(same_class_ids) < n_pairs_pos:
                # Pick another sample
                sample_id = self.rng.randint(0, class_ids[0].size)
                same_class_ids = class_ids[1] == class_ids[1][sample_id]
                same_class_ids[sample_id] = False
      
            # Choose 2 positive and 2 negative pairs
            same_class_ids[sample_id] = False  # make sure our sample is not included
            pos_pairs = class_ids[0][self.rng.choice(np.where(same_class_ids)[0],
                                 size=n_pairs_pos, replace=False)]
            same_class_ids[sample_id] = True # make sure our sample is not included
            neg_pairs = class_ids[0][self.rng.choice(np.where(np.logical_not(same_class_ids))[0],
                                 size=n_pairs_neg, replace=False)]
            
            # Add pairs to the pair_ids
            pair_ids[i:i+n_pairs_pos+n_pairs_neg, 0] = class_ids[0][sample_id]
            pair_ids[i:i+n_pairs_pos, 1] = pos_pairs
            pair_ids[i+n_pairs_pos:i+n_pairs_pos+n_pairs_neg, 1] = neg_pairs
            
            # Add target labels
            pair_labels[i:i+n_pairs_pos] = 1  # positive pairs
            #pair_labels[i+n_pairs_pos:i+n_pairs_pos+n_pairs_neg] = 0  # negative pairs
            
            # Finally, remove the chosen sample from training ids
            class_ids = [np.delete(class_ids[0], sample_id),
                         np.delete(class_ids[1], sample_id)]

        # Store the formed indices
        self.pair_ids = pair_ids
        self.pair_labels = pair_labels          
    
    
    def on_epoch_end(self):
        """Executed after every epoch"""
        self.indexes = np.arange(len(self.pair_labels))
        if self.shuffle:
            self.rng.shuffle(self.indexes)
            
            
    def __len__(self):
        """Get the number of batches per epoch """
        return int(np.floor(len(self.pair_labels)) / self.batch_size)
    
    
    def __getitem__(self, index):
        """Generate one batch of data based on the batch index"""
        # Pre-allocate memory
        X_batch = [np.zeros(self.dim),  # tensor 0
                   np.zeros(self.dim)]  # tensor 1
        y_batch = np.zeros(self.batch_size)
        
        # Batch ids
        batch_ids = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X_ids = self.pair_ids[batch_ids, :]
        y_batch = self.pair_labels[batch_ids]
        
        # Generate batch from current ids
        X_batch[0] = self.spectra[X_ids[:,0], :, :]
        X_batch[1] = self.spectra[X_ids[:,1], :, :]      
        
        return X_batch, y_batch
