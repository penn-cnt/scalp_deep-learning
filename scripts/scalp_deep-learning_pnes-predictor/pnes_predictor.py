import argparse
import pylab as PLT
from sys import exit

# Local imports
from preproc.clean import *
from validation.data_validation import *
from deep_learning.simple_mlp import mlp_handler as MLPH
from deep_learning.tuned_mlp import tuned_mlp_handler as TMLPH
from deep_learning.tuned_ss_mlp import tuned_mlp_handler as TMLPH

if __name__ == '__main__':
    
    # Command line options needed to obtain data.
    parser = argparse.ArgumentParser(description="iEEG to bids conversion tool.")
    parser.add_argument("--feature_file", type=str, required=True, help="Filepath to the feature file.")
    parser.add_argument("--pivotfile", type=str, help="Intermediate file with pivoted inputs. Can be used for different models like LR, ANOVA, etc. Provide to skip generation.")
    parser.add_argument("--mlpfile", type=str, help="Intermediate file with MLP inputs. Has undergone various transformations to better model the data. Provide to skip generation.")
    parser.add_argument("--mlpdir", type=str, help="Directory to store torch datasets.")
    parser.add_argument("--splitmethod", type=str, default='uid', choices=['raw','uid'], help="Split the data into train/test by number of records (raw) or by subjects (uid)")
    parser.add_argument("--ncpu", type=int, default=2, help="Number of cpus.")
    parser.add_argument("--ntrials", type=int, default=10, help="Number of trials.")
    parser.add_argument("--plotdir", type=str, help="Directory to store plots.")
    parser.add_argument("--logfile", type=str, help="Log file path for hyperparameter testing.")
    parser.add_argument("--raydir", type=str, help="Output directory for ray results.")
    args = parser.parse_args()

    # Initialize the data cleaning class
    DM = data_manager(args.feature_file, args.pivotfile, args.mlpfile, args.splitmethod)

    # Prepare the pivot data
    pivot_DF = DM.pivotdata_prep()

    # Prepare the MLP data
    MLP_objects = DM.mlpdata_prep()

    # Perform some validation
    DV = vector_analysis(pivot_DF,MLP_objects)
    DV.quantile_features()
    #DV.linear_seperability_search(f"{args.plotdir}/sleep_seperability_normalized/")



    #DV.bootstrap_validation(f"{args.plotdir}/bootstrap/")
    #DV.calculate_anova()
    #DV.plot_paired_whisker(f"{args.plotdir}/whisker_plots/")
    #DV.plot_paired_pdf(f"{args.plotdir}/pdf_plots/")
    #DV.plot_vectors(f"{args.plotdir}/mlp_vectors/")

    # Run a simple pre-defined MLP model
    #DL = MLPH(MLP_objects)
    #DL.run_mlp(1e-3,64,[0.5,0.5,1,0.5],verbose=True) # .72,.71 Raw

    # Run a tuned MLP model
    #DL = TMLPH(args.ncpu,args.ntrials,args.logfile,args.raydir)
    #DL.create_config(args.mlpfile)
    #DL.run_ray_tune(args.mlpfile)
    #DL.test_set_config()