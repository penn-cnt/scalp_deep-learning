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
    tuning_group.add_argument("--ntrial", type=int, default=100, help="Number of trials.")
    tuning_group.add_argument("--tune_file", type=str, help="Output file for hyper parameter tuning.")
    tuning_group.add_argument("--raydir", type=str, help="Output folder for ray tuning.")

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
    TUNING_HANDLER = tuning_manager(DL_object, args.ncpu, args.ntrial, args.tune_file, args.raydir)
    
    # Run tuner or a single config model
    if not args.test_config:
        # Perform MLP tuning
        TUNING_HANDLER.make_tuning_config_mlp()
        TUNING_HANDLER.run_ray_tune_mlp()
    else:
        config = {'batchsize':256, 'normorder':'after', 'activation':'tanh', 'lr':1e-5, 'weight':1000}
        config[f"frequency_nlayer"]    = 2
        config[f"frequency_hsize_1"]   = 1.45
        config[f"frequency_hsize_2"]   = 0.90
        config[f"frequency_hsize_3"]   = 0.70
        config[f"frequency_drop_1"]    = 0.20
        config[f"frequency_drop_2"]    = 0.30
        config[f"frequency_drop_3"]    = 0.15
        config[f"time_nlayer"]         = 1
        config[f"time_hsize_1"]        = 0.35
        config[f"time_hsize_2"]        = 0.85
        config[f"time_hsize_3"]        = 0.85
        config[f"time_drop_1"]         = 0.25
        config[f"time_drop_2"]         = 0.35
        config[f"time_drop_3"]         = 0.30
        config[f"categorical_nlayer"]  = 2
        config[f"categorical_hsize_1"] = 1.30
        config[f"categorical_hsize_2"] = 0.75
        config[f"categorical_hsize_3"] = 0.75
        config[f"categorical_drop_1"]  = 0.15
        config[f"categorical_drop_2"]  = 0.30
        config[f"categorical_drop_3"]  = 0.10
        config[f"combined_nlayer"]     = 2
        config[f"combined_hsize_1"]    = 1.35
        config[f"combined_hsize_2"]    = 0.10
        config[f"combined_hsize_3"]    = 0.30
        config[f"combined_drop_1"]     = 0.30
        config[f"combined_drop_2"]     = 0.40
        config[f"combined_drop_3"]     = 0.10

        train_pnes(config, DL_object, debug=args.debug, patient_level=False, directload=True)
        #TP(config, DL_object, debug=args.debug, patient_level=False, directload=True)