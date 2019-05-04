import os
from pathlib import Path
import numpy as np
from src import pre_processing, model
from keras import backend as K
from hyperopt import fmin, tpe, hp, Trials
import pickle
import time
from src.PairGenerator import PairGenerator


############ Keras/Tensorflow randomness ###################
from numpy.random import seed
seed(123)
from tensorflow import set_random_seed
set_random_seed(234)
############################################################


# Path to ESC-50 master directory (change if needed)
ESC50_PATH = Path('ESC-50-master')


# Define a parameter space for hyperopt: 
PRMTRS = {'n_filters':      [16, 32, 64],
          'n_dense':        [125, 250, 500],
          'n_conv_layers':  [1, 2, 3, 4],
          'dropout_dense':  [0.0, 0.25, 0.5],
          'dropout_conv':   [0.0, 0.25, 0.5],
          'dist':           ['l1', 'l2'],
          'loss':           ['mse', 'binary_crossentropy'],
          'neg_ratio':      [1.0, 2.0]}

SPACE = {'n_filters':      hp.choice('n_filters',      PRMTRS['n_filters']),
         'n_dense':        hp.choice('n_dense',        PRMTRS['n_dense']),
         'n_conv_layers':  hp.choice('n_conv_layers',  PRMTRS['n_conv_layers']),
         'dropout_dense':  hp.choice('dropout_dense',  PRMTRS['dropout_dense']),
         'dropout_conv':   hp.choice('dropout_conv',   PRMTRS['dropout_conv']),
         'dist':           hp.choice('dist',           PRMTRS['dist']),
         'loss':           hp.choice('loss',           PRMTRS['loss']),
         'neg_ratio':      hp.choice('neg_ratio',      PRMTRS['neg_ratio']),
         'decay':          hp.loguniform('decay', np.log(1e-7), np.log(1e-5)),
         'learning_rate':  hp.loguniform('learning_rate', np.log(0.0005), np.log(0.01))}
        

def reset_seed():
    seed(123)
    set_random_seed(234)
    
    
# Define an objective function for hyperopt
def objective(params):
    print(params)
    
    # Data generators
    batch_size = 20  # change according to your GPU limitations
    train_generator = PairGenerator(X_train, y_train, seed=1, 
                                  amount_of_pairs=1.0,
                                  neg_pair_ratio=params['neg_ratio'],
                                  batch_size=batch_size)
    
    # Model instance with the given parameters
    mdl = model.create_siamese_model(input_shape=X_train.shape[1:],
                                   n_filters=params['n_filters'],
                                   n_dense=params['n_dense'],
                                   n_conv_layers=params['n_conv_layers'],
                                   dropout_conv=params['dropout_conv'],
                                   dropout_dense=params['dropout_dense'], 
                                   dist=params['dist'],
                                   learning_rate=params['learning_rate'], 
                                   decay=params['decay'],
                                   loss=params['loss'])
    
    # Setup callbacks and train
    patience = 0
    best_so_far = 0.0
    
    for i in range(12):
        print("Iteration: {}".format(i))
        # Train for a single epoch        
        history = mdl.fit_generator(generator=train_generator,
                          epochs=1, verbose=1)  
        
        # Use train oneshot data for validation after each epoch
        score = model.test_oneshot(mdl, X_oneshot_train, y_oneshot_train,
                          visualize=False, save_results=False, seed=2)
        
        # Save best score
        if score > best_so_far:
            best_so_far = score
            patience = 0  # reset patience
        else:
            patience += 1
        
        if patience > 6:
            break
    
    print("Best score: {}".format(best_so_far))

    # Convert best validation score into a minimizible form
    final_score = 1 - best_so_far

    # Try loading already existing score history
    try:
        history = pickle.load(open("checkpoint/history.pkl", "rb"))
    except:
        history = []
    
    # Save updated history
    history.append(1-final_score)
    with open("checkpoint/history.pkl", "wb") as f:
        pickle.dump(history, f, pickle.HIGHEST_PROTOCOL)
    
    del mdl
    
    return final_score
    

def run_trials(trials_step=1, init_max_trials=1):
    """ """
    try: # try to load an already existing trials object
        trials = pickle.load(open("checkpoint/checkpoint.hyperopt", "rb"))
        max_trials = len(trials.trials) + trials_step
        print("Continuing from {} trials".format(len(trials.trials)))
        
    except:  # create a new trials object
        print("Starting trials from scratch")
        trials = Trials()
        max_trials = init_max_trials
    
    # Run hyperopt    
    best = fmin(objective, SPACE, algo=tpe.suggest, trials=trials, max_evals=max_trials)
        
    # Save parameters
    with open('checkpoint/params.pkl', 'wb') as f:
        pickle.dump(best, f, pickle.HIGHEST_PROTOCOL)    
        
    # Save current trials object
    with open('checkpoint/checkpoint.hyperopt', 'wb') as f:
        pickle.dump(trials, f, pickle.HIGHEST_PROTOCOL)    


if __name__ == '__main__':   
    # Load inputs, targets and metadata
    X, y, meta = pre_processing.get_data(ESC50_PATH, FULL_DATA=True)    
    
    # Split data into training, oneshot training and oneshot validation sets
    X, X_oneshot_val, y, y_oneshot_val = pre_processing.class_split(X, y,
                                                                   split_size=0.9,
                                                                   seed=1)
    
    X_train, X_oneshot_train, y_train, y_oneshot_train = pre_processing.class_split(X, y,
                                                                       split_size=0.9,
                                                                       seed=3)
    
    X,y = None, None  # release memory
    
    # Addinitional axis for 2D Conv
    X_train = X_train[:,:,:,np.newaxis]
    X_oneshot_val = X_oneshot_val[:,:,:,np.newaxis]
    X_oneshot_train = X_oneshot_train[:,:,:,np.newaxis]
    
    # Run hyperopt
    for i in range(1):
        K.clear_session()
        reset_seed()
        start = time.time()
        run_trials()  
        print("\n\nExecution time: {:.1f} minutes".format((time.time() - start) / 60))

    # Clear tmp folder
    for file in os.listdir(Path('tmp')):
        os.remove(Path('tmp') / file)
