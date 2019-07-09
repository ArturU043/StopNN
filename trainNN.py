'''
Train the Neural Network
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from keras.optimizers import Adam, Nadam
import time
import keras
import pandas
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, AlphaDropout
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from commonFunctions import getYields, FullFOM, myClassifier, gridClassifier, getDefinedClassifier, assure_path_exists
#from scipy.stats import ks_2samp
import localConfig as cfg
import pickle
from prepareDATA import *

if __name__ == "__main__":
    import argparse
    import sys

    ## Input arguments. Pay speciall attention to the required ones.
    parser = argparse.ArgumentParser(description='Process the command line options')
    parser.add_argument('-z', '--batch', action='store_true', help='Whether this is a batch job, if it is, no interactive questions will be asked and answers will be assumed')
    parser.add_argument('-v', '--verbose', action='store_true', help='Whether to print verbose output')
    parser.add_argument('-l', '--layers', type=int, required=True, help='Number of layers')
    parser.add_argument('-n', '--neurons', type=int, required=True, help='Number of neurons per layer')
    parser.add_argument('-e', '--epochs', type=int, required=True, help='Number of epochs')
    parser.add_argument('-a', '--batchSize', type=int, required=True, help='Batch size')
    parser.add_argument('-b', '--learningRate', type=float, required=True, help='Learning rate')
    parser.add_argument('-c', '--decay', type=float, default=0, help='Learning rate decay')
    parser.add_argument('-d', '--dropoutRate', type=float, default=0, help='Drop-out rate')
    parser.add_argument('-r', '--regularizer', type=float, default=0, help='Regularizer')
    parser.add_argument('-i', '--iteration', type=int, default=1, help='Iteration number i')

    args = parser.parse_args()

    n_layers = args.layers
    n_neurons = args.neurons
    n_epochs = args.epochs
    batch_size = args.batchSize
    learning_rate = args.learningRate
    my_decay = args.decay
    dropout_rate = args.dropoutRate
    regularizer = args.regularizer
    iteration = args.iteration

    verbose = 0
    if args.verbose:
        verbose = 1

    ## Model compile arguments, training parameters and optimizer.
    compileArgs = {'loss': 'binary_crossentropy', 'optimizer': 'adam', 'metrics': ["accuracy"]}
    trainParams = {'epochs': n_epochs, 'batch_size': batch_size, 'verbose': verbose}
    myOpt = Adam(lr=learning_rate)#, decay=my_decay)
    compileArgs['optimizer'] = myOpt

    name = "L"+str(n_layers)+"_N"+str(n_neurons)+"_E"+str(n_epochs)+"_Bs"+str(batch_size)+"_Lr"+str(learning_rate)+"_De"+str(my_decay)+"_Dr"+str(dropout_rate)+"_L2Reg"+str(regularizer)+"_Tr"+train_DM+"_Te"+test_point+"_DT"+suffix

    if iteration > 0:
        name = name+"_Ver"+str(iteration)

    ## Directory to save your NN files. Edit lgbk variable in localConfig.py
    filepath = cfg.lgbk+"SingleNN/"+name

    if os.path.exists(filepath) == False:
        os.mkdir(filepath)
    os.chdir(filepath)

    if args.verbose:
        print("Dir "+filepath+" created.")
        print("Starting the training")
        start = time.time()

    ## EXERCISE 2: Create your NN model
    model = # Please, inset your code here........


    ##push test


    ##Fit your model
    history = model.fit(XDev, YDev, validation_data=(XVal,YVal,weightVal), sample_weight=weightDev,shuffle=True, **trainParams)

    acc = history.history["acc"]
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    assure_path_exists(filepath+"/accuracy/"+"dummy.txt")
    assure_path_exists(filepath+"/loss/"+"dummy.txt")

    ## Save accuracy and loss values in a pickle file for later plotting
    pickle.dump(acc, open("accuracy/acc_"+name+".pickle", "wb"))
    pickle.dump(loss, open("loss/loss_"+name+".pickle", "wb"))
    pickle.dump(val_acc, open("accuracy/val_acc_"+name+".pickle", "wb"))
    pickle.dump(val_loss, open("loss/val_loss_"+name+".pickle", "wb"))

    if args.verbose:
        print("Training took ", time.time()-start, " seconds")

    ## Save your trainned model
    model.save(name+".h5")
    model_json = model.to_json()
    with open(name + ".json", "w") as json_file:
      json_file.write(model_json)
    model.save_weights(name + ".h5")

    if args.verbose:
        print("Getting predictions")

    devPredict = model.predict(XDev)
    valPredict = model.predict(XVal)

    if args.verbose:
        print("Getting scores")

    scoreDev = model.evaluate(XDev, YDev, sample_weight=weightDev, verbose = 0)
    scoreVal = model.evaluate(XVal, YVal, sample_weight=weightVal, verbose = 0)

    if args.verbose:
        print "Calculating FOM:"
    dataVal["NN"] = valPredict

    tmpSig, tmpBkg = getYields(dataVal)
    sigYield, sigYieldUnc = tmpSig
    bkgYield, bkgYieldUnc = tmpBkg

    sigDataVal = dataVal[dataVal.category==1]
    bkgDataVal = dataVal[dataVal.category==0]

    fomEvo = []
    fomCut = []

    for cut in np.arange(0.0, 0.9999999, 0.001):
      sig, bkg = getYields(dataVal, cut=cut)
      if sig[0] > 0 and bkg[0] > 0:
        fom, fomUnc = FullFOM(sig, bkg)
        fomEvo.append(fom)
        fomCut.append(cut)

    max_FOM=0

    if args.verbose:
        print "Maximizing FOM"

    for k in fomEvo:
      if k>max_FOM:
        max_FOM=k
    if args.verbose:
        print "Signal@Presel:", sigDataVal.weight.sum() * 35866 * 2
        print "Background@Presel:", bkgDataVal.weight.sum() * 35866 * 2
        print "Signal:", sigYield, "+-", sigYieldUnc
        print "Background:", bkgYield, "+-", bkgYieldUnc

        print "Maximized FOM:", max_FOM
        print "FOM Cut:", fomCut[fomEvo.index(max_FOM)]

    ## Plot accuracy and loss evolution over epochs for both training and validation datasets
    if not args.batch:
        import sys
        import matplotlib.pyplot as plt

        ## Plot accuracy for training and validation datasets of epochs
        # Please, inset your code here........

        ## Plot loss for training and validation datasets of epochs
        # Please, inset your code here........

        ## Save your plots
        # Please, inset your code here........

        if args.verbose:
            print "Model name: "+name
        sys.exit("Done!")
