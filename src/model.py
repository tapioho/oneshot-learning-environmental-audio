import numpy as np
from pathlib import Path
from src.pre_processing import __update_progressbar
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, BatchNormalization, Activation, Conv2D, MaxPooling2D
from keras.layers import Input, Lambda
from keras.optimizers import Adam
import matplotlib.pyplot as plt


def create_siamese_model(input_shape, 
                       n_filters=64, n_dense=500,
                       n_conv_layers=3, n_dense_layers=1, 
                       dist='l2', 
                       dropout_dense=0.25, dropout_conv=0.25,
                       learning_rate=0.001, decay=1e-6,
                       loss='binary_crossentropy'):
    """Create a siamese network with the given parameters. 
       Default parameters are chosen based on parameter_optimization.py"""
    
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    
    # CNN for each input
    cnn = Sequential()
    cnn.add(BatchNormalization())
    
    # Convolutional layers
    for i in range(n_conv_layers):
        cnn.add(Conv2D(n_filters, (5,5), 
                       padding='valid'))
        cnn.add(BatchNormalization())
        cnn.add(Activation('relu'))
        cnn.add(MaxPooling2D(pool_size=(2,2), 
                             padding='Valid'))
        cnn.add(Dropout(dropout_conv))

    # Dense layers
    cnn.add(Flatten())
    for i in range(n_dense_layers):
        cnn.add(Dense(n_dense))
        cnn.add(BatchNormalization())
        cnn.add(Activation('sigmoid'))
        cnn.add(Dropout(dropout_dense))
    
    # Use CNN to encode each input into a tensor
    left_encoded = cnn(left_input)
    right_encoded = cnn(right_input)
    
    # Use the squared/absolute difference between encoded inputs to merge them
    if dist=='l1':
        dist_layer = Lambda(lambda tensors: K.abs(tensors[0]-tensors[1]))
    elif dist=='l2':
        dist_layer = Lambda(lambda tensors: K.square(tensors[0]-tensors[1]))
    else:
        raise ValueError("Invalid distance parameter: {}".format(dist))
        
    inputs_merged = dist_layer([left_encoded, right_encoded])
    
    # Final layer
    prediction = Dense(1, activation='sigmoid')(inputs_merged)
    
    # Model instance 
    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)  
    
	# Optimizer and compile
    opt = Adam(lr=learning_rate, decay=decay)
    siamese_net.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
    
    return siamese_net
    

def test_oneshot(mdl, X_oneshot, y_oneshot, visualize=True, save_results=True, seed=None):
    """Test one-shot classification for each unseen class
        Inputs:
            mdl:
            X_oneshot: Unseen spectra used for one-shot classification
            y_oneshot: Corresponding labels 
            visualize: Show results in a bar plot and a confusion matrix
            save_results: Save results to 'results' folder
            seed: set the random seed to a known value
        Outputs:
            mean score: mean one-shot accuracy over all classes
            """
    
    # Set seed if specified
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    
    # Load class names for decoding
    class_names = np.load(Path('tmp') / 'classes.npy')
    
    # Amount of tests for each sample
    n_tests = 20
    
    # Find the classes which are used for oneshot classification
    classes = np.where(np.sum(y_oneshot, axis=0))[0]
    n_classes = len(classes)
    class_names = class_names[classes]
    
    # Pre-allocate memory
    samples_to_classify = np.zeros(n_tests, dtype=int)
    samples_to_compare = np.zeros((n_tests, n_classes), dtype=int)
    confusion_mat = np.zeros((classes.size, classes.size))
    pairs = [np.zeros((classes.size, X_oneshot.shape[1], X_oneshot.shape[2], X_oneshot.shape[3])),
            np.zeros((classes.size, X_oneshot.shape[1], X_oneshot.shape[2], X_oneshot.shape[3]))]    

    # Initalize accuracy measurements
    correct_positives = np.zeros(classes.size)
    
    # Test model on all of the classes
    shuffle_ids = np.arange(40)
    rng.shuffle(shuffle_ids)
    print("Evaluating one-shot performance...")
    for i in range(n_classes):
        # Divide current class samples into two, half for classification and half for comparing
        samples = np.where(y_oneshot[:,classes[i]])[0]
        
        samples_to_compare[:,i] = samples[shuffle_ids[:n_tests]]
        samples_to_classify[:] = samples[shuffle_ids[n_tests:]]
        
        # Pick n_test amount of other classes for comparing
        for j in range(n_classes):
            if j != i:
                samples = np.where(y_oneshot[:,classes[j]])[0]                                
                samples_to_compare[:,j] = samples[shuffle_ids[:n_tests]]

        # Make predictions by comparing a sample from each class to the sample under classification
         
        # Perform tests
        for k in range(samples_to_classify.size):
            for j in range(samples_to_compare.shape[0]):
                # Form input pairs
                pairs[0][:,:,:,:] = np.repeat(np.expand_dims(X_oneshot[samples_to_classify[k],:,:,:], axis=0), 
                                             classes.size, axis=0)
                pairs[1][:,:,:,:] = X_oneshot[samples_to_compare[j,:], :, :,:]
                
                # Make predictions
                preds = mdl.predict(pairs)
                
                # Add the most similar to confusion matrix
                confusion_mat[i,np.argmax(preds)] += 1
                            
                # Classify based on the most similar 
                if np.argmax(preds) == i:
                    correct_positives[i] += np.count_nonzero(preds[0])
                    
            __update_progressbar((i)/classes.size + 
                                ((k+1)/samples_to_classify.size)/classes.size)
            
        __update_progressbar((i+1)/classes.size)
    
    # Convert to percentage
    correct_positives /= n_tests**2
    
    # Save results
    if save_results:
        np.save(Path('results/confusion_matrix.npy'), confusion_mat)
        np.save(Path('results/class_names.npy'), class_names)
        np.save(Path('results/correct_positives.npy'), correct_positives)
    
    if visualize:
        # Plot results for each class
        print("\nOne-shot classification results for {} classes with 400 classifications for each class:".format(classes.size))        
        __plot_oneshot(correct_positives, labels=class_names)
        __plot_confusion_mat(confusion_mat, class_names)
        
    print("\nScore: {:.2f} \n".format(np.mean(correct_positives)))
    return np.mean(correct_positives)


def __plot_oneshot(accuracies, labels):
    """Plot a bar plot representing the results"""
    x = np.arange(accuracies.size)
    plt.bar(x, accuracies)
    plt.ylim(0.0, 1.0)
    plt.ylabel('Accuracy')
    plt.xticks(x,labels, rotation='vertical')
    plt.title("Mean accuracy: {:.3f}".format(np.mean(accuracies)))
    plt.show()


def __plot_confusion_mat(cm, classes):
    """Plot a confusion matrix representing the classification results """
    title = "Predictions"

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    #classes = unique_labels(y_test)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])),
                    ha="center", va="center",
                    color="black" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    plt.show()
