import json
import argparse
import pylab as PLT
from sys import exit

# Local imports
from preproc.clean import *
from deep_learning.mlp_tuner import *

if __name__ == '__main__':
    
    # Command line options needed to obtain data.
    parser = argparse.ArgumentParser(description="iEEG to bids conversion tool.")
    parser.add_argument("--feature_file", type=str, required=True, help="Filepath to the feature file.")
    parser.add_argument("--target_col", type=str, default='target_epidx', help="Target vector to model.")

    pivot_group = parser.add_argument_group('Pivot Options')
    pivot_group.add_argument("--pivot_file", type=str, help="Intermediate file with pivoted inputs. Can be used for different models like LR, ANOVA, etc. Provide to skip generation.")
    pivot_group.add_argument("--clip_length", type=int, default=30, help="Expected clip length for the study. Used to remove other clips used for baseline stats (like Marsh)")
    pivot_group.add_argument("--marsh_threshold", type=float, default=2.0, help="Threshold for marsh criterion. Original marsh=2, lower removes more outliers, larger allows more outliers.")

    vector_group = parser.add_argument_group('DL Vector Options')
    vector_group.add_argument("--vector_file", type=str, help="Intermediate file with DL vectors.")
    vector_group.add_argument("--criteria_file", type=str, help="Optional. Yaml file with criteria for data usage in model.")
    vector_group.add_argument("--mapping_file", type=str, default='./configs/mappings.yaml', help="Yaml file with column mappings.")
    vector_group.add_argument("--transformer_file", type=str, default='./configs/transformer_blocks.yaml', help="Yaml file with criteria for vector transformer blocks.")
    vector_group.add_argument("--vector_plot_dir", type=str, help="Optional. Directory to save plots of the input vector distributions.")

    tuning_group = parser.add_argument_group('DL Tuning Options')
    tuning_group.add_argument("--ncpu", type=int, default=1, help="Number of cpus to use for hyperparameter tuning.")
    tuning_group.add_argument("--ntrial", type=int, default=100, help="Number of trials.")
    tuning_group.add_argument("--tune_file", type=str, help="Output file for hyper parameter tuning.")
    tuning_group.add_argument("--raydir", type=str, help="Output folder for ray tuning.")

    checkpoint_group = parser.add_argument_group('Checkpoint options')
    checkpoint_group.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint for model.")

    method_group = parser.add_mutually_exclusive_group()
    method_group.add_argument("--test_config", action='store_true', default=False, help="Run model using just one config setting. Good for testing.")
    method_group.add_argument("--test_model", action='store_true', default=False, help="Run model using saved torch model.")
    method_group.add_argument("--raytune", action='store_true', default=False, help="Use raytume.")
    method_group.add_argument("--fit_patient", action='store_true', default=False, help="Fit new patient data.")

    misc_group = parser.add_argument_group('Misc Options')
    misc_group.add_argument("--patient_level", action='store_true', default=False, help="Fit patient level arrays.")
    misc_group.add_argument("--debug", action='store_true', default=False, help="Show extra debugging info.")
    misc_group.add_argument("--track_weights", action='store_true', default=False, help="Track if the model is updating weights at each epoch.")
    misc_group.add_argument("--outdir", type=str, default='/Users/bjprager/Documents/GitHub/scalp_deep-learning/user_data/derivative/MODELS/', help="Output directory for any miscelaneous info.")

    args = parser.parse_args()

    # Initialize the pivot workflow
    PM       = pivot_manager(args.feature_file, args.pivot_file, args.clip_length)
    pivot_DF = PM.workflow()

    # Make the DL vectors
    VM        = vector_manager(pivot_DF, args.target_col, args.vector_file, args.criteria_file, args.mapping_file, args.transformer_file, args.vector_plot_dir)
    DL_object = VM.workflow()

    # Initialize the ray tuning class
    if args.checkpoint == None:
        config = json.load(open("configs/base_config.json",'r'))
    else:
        config = torch.load(args.checkpoint)['config']

    # Run tuner or a single config model
    if args.raytune:
        # Perform MLP tuning
        TUNING_HANDLER = tuning_manager(DL_object, args.ncpu, args.ntrial, args.tune_file, args.raydir, config, args.patient_level)
        TUNING_HANDLER.make_tuning_config_mlp()
        TUNING_HANDLER.run_ray_tune_mlp()
    elif args.test_config:
        train_pnes_handler(config, DL_object, patient_level=args.patient_level, raytuning=False)
    elif args.test_model:
        train_pnes_handler(config, DL_object, patient_level=args.patient_level, raytuning=False, checkpoint_path=args.checkpoint)
    elif args.fit_patient:
        update_pnes_handler(config, DL_object, checkpoint_path=args.checkpoint)