import argparse
import numpy as np
import pandas as PD
from sys import argv
from tqdm import tqdm

class clean_yasa:
    """
    Clean up the yasa feature output to display results on thirty second windows.
    """

    def __init__(self,yasa_path,yasa_window_size):

        self.raw_yasa         = PD.read_pickle(yasa_path)
        self.yasa_window_size = yasa_window_size

    def pipeline(self):

        self.data_prep()
        self.format_predictions()
        self.make_dataframe()
        return self.outDF

    def data_prep(self):
        """
        Only grab segments with the minimum time requirement and get the metadata out.
        """

        # Only allow clips that had five minutes of data
        inds          = (self.raw_yasa['t_end'].values-self.raw_yasa['t_start'].values)>=300
        self.raw_yasa = self.raw_yasa.iloc[inds]

        # Make the lookup column list and get channel names
        lookup_cols    = ['file', 't_start', 't_end', 't_window', 'method', 'tag']
        self.channels  = np.setdiff1d(self.raw_yasa.columns,lookup_cols)

        # get the column headers for predictions
        self.yasa_cols = np.array(self.raw_yasa.tag.values[0].split(','))

    def format_predictions(self):
        """
        Format the predictions to be a time by yasa channel array
        """

        # Get the YASA prediction. Which should be the same for all channels as we use a consensus across channels
        self.predictions = []
        for idx,ival in enumerate(self.raw_yasa[self.channels[0]].values):
            
            # Get the expected prediction shape
            try:
                formatted_pred = ival.replace('|',',')
                formatted_pred = np.array(formatted_pred.split(',')).reshape((-1,self.yasa_cols.size))
                self.predictions.append(formatted_pred)
            except:
                nrow = np.floor(self.yasa_window_size/30).astype('int')
                self.predictions.append(np.nan*np.ones((nrow,self.yasa_cols.size)))

    def make_dataframe(self):
        """
        Make the predictions into a similar format as the feature dataframe for easier merging.
        """

        # Get the start time and filename for each row
        files   = self.raw_yasa.file.values
        t_start = self.raw_yasa.t_start.values
        
        # Make the final lookup tables
        outfile  = []
        outstart = []
        outend   = []
        outstage = []
        for idx,ifile in tqdm(enumerate(files),total=len(files)):
            istart = t_start[idx]
            ipred  = self.predictions[idx]
            for jdx,sleep_stage in enumerate(ipred):
                outfile.append(ifile)
                outstart.append(istart+(jdx*30))
                outend.append(istart+((jdx+1)*30))
                outstage.append(sleep_stage)
        outstage = np.array(outstage)

        # Make the lookup dataframe
        outDF = PD.DataFrame(outfile,columns=['file'])
        outDF['t_start']          = outstart
        outDF['t_end']            = outend
        for idx,icol in enumerate(self.yasa_cols):
            outDF[f"yasa_{icol}"] = outstage[:,idx]
        
        # Sort the results
        self.outDF = outDF.sort_values(by=['file','t_start'])

class merge_yasa:

    def __init__(self,feature_path,yasa_df):

        self.yasa     = yasa_df
        self.features = PD.read_pickle(feature_path) 

    def pipeline(self):
        
        self.yasa_mapping()
        self.joint_prediction()
        self.merge_results()

    def yasa_mapping(self):
        """
        Apply project specific mapping to the yasa labels.
        """

        # Clean up the labels to just be sleep or wake
        new_map             = {'N1':'S','N2':'S','N3':'S','R':'S','W':'W'}
        self.consensus_cols = [icol for icol in self.yasa if 'yasa' in icol]
        for icol in self.consensus_cols:
            self.yasa[icol] = self.yasa[icol].apply(lambda x: new_map[x] if x in new_map.keys() else 'U')

    def joint_prediction(self):
        """
        Get the joint prediction for yasa
        """

        # Get the consensus prediction
        preds                       = self.yasa[self.consensus_cols].mode(axis=1).values
        self.yasa['yasa_consensus'] = preds.flatten()
        self.yasa                   = self.yasa.drop(self.consensus_cols,axis=1)

    def merge_results(self):

        # Create the yasa lookup arrays
        yasa_files     = self.yasa.file.values
        yasa_tstart    = self.yasa.t_start.values
        yasa_tend      = self.yasa.t_end.values
        unique_files   = np.unique(yasa_files)

        # Create the feature dataframe lookup arrays
        feature_files  = self.features.file.values
        feature_tstart = self.features.t_start.values

        # Populate the YASA feature column with unknowns that we can replace by index with the correct value
        YASA_FEATURE = np.array(['U' for ii in range(self.features.shape[0])])
        YASA_LOOKUP  = self.yasa['yasa_consensus'].values
        
        # Step through the unique files
        for ifile in tqdm(unique_files,total=unique_files.size):
                    
            # Get the file indices
            yasa_file_inds    = (yasa_files==ifile)
            feature_file_inds = (feature_files==ifile)
            
            # The yasa lookup was made for more than just the PNES project. So we can cull for files in the feature df
            if feature_file_inds.sum() > 0:

                # Step through the time values
                unique_tstart = np.unique(yasa_tstart[yasa_file_inds])
                for istart in unique_tstart:

                    # Get the time indices
                    yasa_time_inds    = (yasa_tstart==istart)
                    feature_time_inds = (feature_tstart>=istart)&(feature_tstart<(istart+30))

                    # Get the current prediction, if available
                    YASA_slice = YASA_LOOKUP[yasa_file_inds&yasa_time_inds]

                    # Step through the possible outcomes for the yasa slice size
                    combined_inds = feature_file_inds&feature_time_inds
                    if combined_inds.sum() > 0:
                        if YASA_slice.size == 1:
                            YASA_FEATURE[combined_inds] = YASA_slice[0]
                        elif YASA_slice.size > 1:
                            raise Exception("Too many YASA values map to this feature. Check YASA generation.")
                        else:
                            pass
        self.YASA_FEATURE = YASA_FEATURE
        print(self.YASA_FEATURE)

if __name__ == '__main__':

    # Command line options needed to obtain data.
    parser   = argparse.ArgumentParser()
    parser.add_argument('--feature_path', type=str, help='Input path to the feature dataframe.')
    parser.add_argument('--yasa_path', type=str, help='Input path to the yasa dataframe.')
    parser.add_argument('--yasa_window_size', type=int, default=300, help='Input path to the yasa dataframe.')
    args = parser.parse_args()

    # Clean up the yasa data
    CLN    = clean_yasa(args.yasa_path,args.yasa_window_size)
    yasaDF = CLN.pipeline()

    # Merge the results
    MRG = merge_yasa(args.feature_path,yasaDF)
    MRG.pipeline()