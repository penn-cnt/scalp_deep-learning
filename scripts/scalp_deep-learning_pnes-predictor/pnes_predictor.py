import argparse
import pylab as PLT
from sys import exit

# Local imports
from preproc.clean import *
from deep_learning.mlp_tuner import *
from nntest.basic_nn import train_pnes as TP

if __name__ == '__main__':
    
    # Command line options needed to obtain data.
    parser = argparse.ArgumentParser(description="iEEG to bids conversion tool.")
    parser.add_argument("--feature_file", type=str, required=True, help="Filepath to the feature file.")
    parser.add_argument("--target_col", type=str, default='target_epidx', help="Target vector to model.")

    pivot_group = parser.add_argument_group('Pivot Options')
    pivot_group.add_argument("--pivot_file", type=str, help="Intermediate file with pivoted inputs. Can be used for different models like LR, ANOVA, etc. Provide to skip generation.")
    pivot_group.add_argument("--clip_length", type=int, default=30, help="Expected clip length for the study. Used to remove other clips used for baseline stats (like Marsh)")

    vector_group = parser.add_argument_group('DL Vector Options')
    vector_group.add_argument("--vector_file", type=str, help="Intermediate file with DL vectors.")
    vector_group.add_argument("--criteria_file", type=str, help="Optional. Yaml file with criteria for data usage in model.")
    vector_group.add_argument("--mapping_file", type=str, default='./configs/mappings.yaml', help="Yaml file with column mappings.")
    vector_group.add_argument("--transformer_file", type=str, default='./configs/transformer_blocks.yaml', help="Yaml file with criteria for vector transformer blocks.")
    vector_group.add_argument("--vector_plot_dir", type=str, help="Optional. Directory to save plots of the input vector distributions.")

    tuning_group = parser.add_argument_group('DL Tuning Options')
    tuning_group.add_argument("--ncpu", type=int, default=1, help="Number of cpus to use for hyperparameter tuning.")
    tuning_group.add_argument("--tune_file", type=str, help="Output file for hyper parameter tuning.")

    misc_group = parser.add_argument_group('Misc Options')
    misc_group.add_argument("--debug", action='store_true', default=False, help="Show extra debugging info.")
    misc_group.add_argument("--test_config", action='store_true', default=False, help="Run model using just one config setting. Good for testing.")
    misc_group.add_argument("--track_weights", action='store_true', default=False, help="Track if the model is updating weights at each epoch.")

    args = parser.parse_args()

    # Initialize the pivot workflow
    PM       = pivot_manager(args.feature_file, args.pivot_file, args.clip_length)
    pivot_DF = PM.workflow()

    # Make the DL vectors
    VM        = vector_manager(pivot_DF, args.target_col, args.vector_file, args.criteria_file, args.mapping_file, args.transformer_file, args.vector_plot_dir)
    DL_object = VM.workflow()

    # Initialize the ray tuning class
    TUNING_HANDLER = tuning_manager(DL_object,args.ncpu,args.tune_file)
    
    # Run tuner or a single config model
    if not args.test_config:
        # Perform MLP tuning
        TUNING_HANDLER.make_tuning_config_mlp()
        TUNING_HANDLER.run_ray_tune_mlp()
    else:
        config = {'batchsize':128,'normorder':'first','activation':'relu','lr':1e-2}
        config['frequency']   = {'nlayer':3,'hsize_1':0.9,'hsize_2':0.8,'hsize_3':0.6,'drop_1':0.6,'drop_2':0.4,'drop_3':0.2}
        config['time']        = {'nlayer':2,'hsize_1':0.8,'hsize_2':0.6,'drop_1':0.6,'drop_2':0.4}
        config['categorical'] = {'nlayer':1,'hsize_1':0.6,'drop_1':0.6}
        config['combined']    = {'nlayer':1,'hsize_1':0.2,'drop_1':0.6}
        train_pnes(config, DL_object, debug=args.debug, patient_level=False, directload=True)
        #TP(config, DL_object, debug=args.debug, patient_level=False, directload=True)