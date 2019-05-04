import os
from pathlib import Path
import numpy as np
from src import pre_processing, model
from src.PairGenerator import PairGenerator
from keras import backend as K
from keras.models import load_model


############ Keras/Tensorflow randomness ###################
from numpy.random import seed
seed(123)
from tensorflow import set_random_seed
set_random_seed(234)
############################################################


# Path to ESC-50 master directory (change if needed)
ESC50_PATH = Path('ESC-50-master')


if __name__ == '__main__':
    # Load inputs, targets and metadata
    X, y, meta = pre_processing.get_data(ESC50_PATH, FULL_DATA=True)    
    X, X_oneshot_test, y, y_oneshot_test = pre_processing.class_split(X,y,split_size=0.9, seed=1)
    X_train, X_oneshot_val, y_train, y_oneshot_val = pre_processing.class_split(X,y,split_size=0.9, seed=3)   
    X,y = None, None  # release memory
    
    # Addinitional axis for 2D Conv
    X_train = X_train[:,:,:,np.newaxis]
    X_oneshot_test = X_oneshot_test[:,:,:,np.newaxis]
    X_oneshot_val = X_oneshot_val[:,:,:,np.newaxis]
    
    # Make sure the session is clear
    K.clear_session()
    
    # Create a data generator
    batch_size = 20  # change according to your GPU limitations
    pairs = 1.0  # amount of pairs from the maximum
    neg_pair_ratio = 1.0  # negative-to-positive-pair ratio
    train_generator = PairGenerator(X_train, y_train, seed=1, 
                              amount_of_pairs=pairs, neg_pair_ratio=neg_pair_ratio,
                              batch_size=batch_size)    
    
    checkpoint_path = Path('checkpoint')
    
    try:
       mdl = load_model(str(checkpoint_path / 'model.h5'))
       print("Continuing training")
    except:
        print("Starting training from scratch")
        # Create a siamese network model and fit
		# Default parameters are chosen based on the hyperparameter 
		# optimization process
        mdl = model.create_siamese_model(input_shape=X_oneshot_val.shape[1:])
									   
    # Setup callbacks    
    patience = 0
    patience_thresh = 10
    best_so_far = 0.0
	
    for i in range(20):
        print("Iteration: {}".format(i))        
        history = mdl.fit_generator(generator=train_generator,
                          epochs=1, verbose=1)  
        
        score = model.test_oneshot(mdl, X_oneshot_val, y_oneshot_val,
                          visualize=False, save_results=False, seed=1)
        
        # Save best score
        if score > best_so_far:
            mdl.save_weights(checkpoint_path / 'best_weights.h5')
            best_so_far = score
            patience = 0
        else:
            patience += 1
        
        if patience > patience_thresh:
            break
        
    # Load best weights
    mdl.load_weights(checkpoint_path / 'best_weights.h5')
    
    # Perform oneshot test
    print("One-shot test")
    model.test_oneshot(mdl, X_oneshot_test, y_oneshot_test,
                          visualize=True, save_results=True, seed=1)
    mdl.save(str(checkpoint_path / 'model.h5'))  # save for future training

    # Clear tmp folder
    for file in os.listdir(Path('tmp')):
        os.remove(Path('tmp') / file)
    del mdl
    K.clear_session()
    
    
    
