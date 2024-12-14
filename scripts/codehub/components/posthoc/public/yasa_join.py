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

    print(yasaDF)