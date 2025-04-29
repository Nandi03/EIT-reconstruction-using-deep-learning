from eit_reconstruction_models import EITReconstructionModel as EITModel
import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import time
from load_eit_data_from_csv import DataLoader
from classification_report_metrics import *

def add_gaussian_noise(X, mean=0.0, std=0.01):
    noise = np.random.normal(loc=mean, scale=std, size=X.shape)
    return X + noise

config = None
with open("configuration.yaml", "r") as f:
    config = yaml.safe_load(f)
 
num_runs = config['num_runs']
num_epochs = 2000  
scaler_type = config['scaler']

mode = config['mode']
variations = config["configurations"]
network = config['network']

# Initialise a dictionary to store metrics for all configurations
all_metrics = {}

if mode == "classification":
    j_accs = np.zeros(num_runs)
    ham_loss = np.zeros(num_runs)
    f1_scores = np.zeros(num_runs)
    rmses = np.zeros(num_runs)
else:
    mses = np.zeros(num_runs)
    maes = np.zeros(num_runs)
    rmses = np.zeros(num_runs)


for configuration in variations:

    # read configuration file
    skin_type = configuration["skin_type"]
    num_voltages = configuration["num_voltages"]
    dir_name = configuration["dir"]
    single_touch_file = configuration["single_touch_file"]
    multi_touch_file = configuration["multi_touch_file"]
    pickle_fname = configuration["pickle_fname"]
    if mode == "force_predictions":
        pickle_fname += "_with_force"
    elif mode == "random_force_predictions":
        pickle_fname += "_with_randforce"

    # intialise parameters
    best_seed = 42  
    best_f1_score = float('-inf')
    best_rmse = float('+inf')
    three_d = False if skin_type == "2d" else True
    force = True if mode == "force_predictions" else False
    rand_force = True if mode == "rand_force_predictions" else False

    # load data
    dataloader = DataLoader(num_voltages)
    dataloader.load_data(os.path.join(".", dir_name, single_touch_file), 
                            os.path.join(".", dir_name, multi_touch_file), 
                            three_d=three_d, force=force, randforce=rand_force)
    
    print(f"Training for configuration: {pickle_fname}")
    for r in range(num_runs):
        print(f"Run: {r + 1}")
        
        scaler = StandardScaler()

        # initialise model class
        model_cls = EITModel(num_voltages, len(dataloader.index_to_coordinate))
        
        # store index_to_coordinate dict for plotting later
        with open(f"{pickle_fname}_index_to_coordinate.pkl", "wb") as f:
            pickle.dump(dataloader.index_to_coordinate, f)
        
        # initialise model
        if mode == "classification":
            if network == "dnn":
                model = model_cls.create_dnn_model_classification()
            else:
                model = model_cls.create_1d_cnn_model_classification()
        else:
            if network == "cnn":
                model = model_cls.create_dnn_model_regression()
            else:
                model = model_cls.create_1d_cnn_model_regression()
        
        # data split
        X_train, X_test, y_train, y_test = train_test_split(dataloader.voltage_array, 
                                                            dataloader.output_array, 
                                                            test_size=0.4, random_state=42 + r)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, 
                                                        test_size=0.5, random_state= 42 + r)

        # add noise to data
        X_train = add_gaussian_noise(X_train)

        # scale data
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        X_val = scaler.transform(X_val)
        
        # train model for current run
        history = model_cls.model.fit(X_train, y_train, 
                    validation_data=(X_val, y_val),
                    epochs=num_epochs,
                    batch_size=128, 
                    callbacks=[], verbose=0)
        
        # evaluation
        print(f"Evaluating run {r + 1}")
        if mode == "classification":
            _, j_accs[r], ham_loss[r], f1_scores[r], rmses[r] = model_cls.model.evaluate(X_test, y_test)
            if f1_scores[r] > best_f1_score:
                best_f1_score = f1_scores[r]
                best_seed = 42 + r
        else:
            mses[r], maes[r], rmses[r] = model_cls.model.evaluate(X_test, y_test)
            if rmses[r] < best_rmse:
                best_rmse = rmses[r]
                best_seed = 42 + r

            
    # Print average results
    print(f"For configuration: {pickle_fname}")
            
    # Train best model with timing
    print(f"\nTraining best model for {pickle_fname} with seed {best_seed}")
    start_time = time.time()
    
    # reinitialise and prepare best model
    model_cls = EITModel(num_voltages,  len(dataloader.index_to_coordinate))
    model = None
    if mode == "classification":
        if network == "dnn":
            model = model_cls.create_dnn_model_classification()
        else:
            model = model_cls.create_1d_cnn_model_classification()
    else:
        if network == "cnn":
            model = model_cls.create_dnn_model_regression()
        else:
            model = model_cls.create_1d_cnn_model_regression()
    
    # Data splitting with best seed
    X_train, X_test, y_train, y_test = train_test_split(
        dataloader.voltage_array, dataloader.output_array,
        test_size=0.4, random_state=best_seed
    )
    X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test, test_size=0.5, random_state=best_seed
    )
    
    X_train, X_test, X_val = add_gaussian_noise(X_train), add_gaussian_noise(X_test), add_gaussian_noise(X_val)
    
    # Data scaling
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)
    
    # Training best model
    history = model_cls.model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=num_epochs,
        batch_size=128,
        callbacks=[],
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    # Get all final metrics
    test_metrics = model_cls.model.evaluate(X_test, y_test, verbose=0)
    val_metrics = model_cls.model.evaluate(X_val, y_val, verbose=0)
    
    # Prepare metrics dictionary
    metrics = {
        'best_seed': best_seed,
        'training_time_seconds': training_time,
        'final_epoch': {
            'train_metrics': {k: v[-1] for k, v in history.history.items() if not k.startswith('val_')},
            'val_metrics': {k: v[-1] for k, v in history.history.items() if k.startswith('val_')}
        },
        'test_metrics': dict(zip(model_cls.model.metrics_names, test_metrics)),
        'val_metrics': dict(zip(model_cls.model.metrics_names, val_metrics)),
        'average_metrics': {}
    }
    
    # Add run averages
    if mode == "classification":
        metrics['average_metrics'] = {
            'jaccard_acc': np.mean(j_accs),
            'hamming_loss': np.mean(ham_loss),
            'f1_score': np.mean(f1_scores),
            'rmse': np.mean(rmses)
        }
        metrics['classification_report'] = print_report(
            y_test, (model_cls.model.predict(X_test) >= 0.5).astype(int)
        )
    else:
        metrics['average_metrics'] = {
            'mse': np.mean(mses),
            'mae': np.mean(maes),
            'rmse': np.mean(rmses)
        }
    
    # Save model and metrics
    model_cls.save_model(f"{pickle_fname}_model_{mode}.weights.h5")
    with open(f"{pickle_fname}_metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)
    
    print(f"\nCompleted training for {pickle_fname}")
    print(f"Training time: {training_time:.2f} seconds")
    print("Final test metrics:", metrics['test_metrics'])
