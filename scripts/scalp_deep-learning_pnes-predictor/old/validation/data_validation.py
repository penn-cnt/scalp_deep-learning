import numpy as np
import pylab as PLT
import pandas as PD
from sys import exit
import seaborn as sns
from tqdm import tqdm
from scipy.stats import zscore
from scipy.stats import ttest_ind,ttest_rel
from sklearn.feature_selection import f_classif

class vector_analysis:

    def __init__(self,pivot_data,mlp_object):
        self.data       = pivot_data
        self.mlp_object = mlp_object 

        # Store scaled input vectors
        self.model_block         = self.mlp_object[8]
        self.X_train_bandpower   = PD.DataFrame(self.mlp_object[0],columns=self.model_block['bandpower'])
        self.X_train_timeseries  = PD.DataFrame(self.mlp_object[2],columns=self.model_block['timeseries'])
        self.X_train_categorical = PD.DataFrame(self.mlp_object[4],columns=self.model_block['categoricals'])
        self.X_test_bandpower    = PD.DataFrame(self.mlp_object[1],columns=self.model_block['bandpower'])
        self.X_test_timeseries   = PD.DataFrame(self.mlp_object[3],columns=self.model_block['timeseries'])
        self.X_test_categorical  = PD.DataFrame(self.mlp_object[5],columns=self.model_block['categoricals'])

        # Attach the targets
        target_vector       = np.array(['epilepsy' for ival in self.mlp_object[6].flatten()])
        mask                = self.mlp_object[6].flatten()==0
        target_vector[mask] = 'pnes'
        self.X_train_bandpower['target']   = target_vector
        self.X_train_timeseries['target']  = target_vector
        self.X_train_categorical['target'] = target_vector

        # Attach the localization
        train_uid_localization = mlp_object[9]['uid'].values
        self.X_train_bandpower['uid']   = train_uid_localization
        self.X_train_timeseries['uid']  = train_uid_localization
        self.X_train_categorical['uid'] = train_uid_localization

    def bootstrap_validation(self,outdir):

        # Merge the dataframes
        bandpower  = PD.concat((self.X_train_bandpower,self.X_test_bandpower))
        timeseries = PD.concat((self.X_train_timeseries,self.X_test_timeseries))

        # Define a blacklist of columns to avoid bad pairings
        blacklist        = ['target','uid']
        bandpower_fcols  = np.setdiff1d(bandpower.columns,blacklist)
        timeseries_fcols = np.setdiff1d(timeseries.columns,blacklist)

        # Break out by target
        bandpower_pnes      = bandpower.loc[bandpower.target=='pnes']
        bandpower_epilepsy  = bandpower.loc[bandpower.target=='epilepsy']
        timeseries_pnes     = timeseries.loc[timeseries.target=='pnes']
        timeseries_epilepsy = timeseries.loc[timeseries.target=='epilepsy']

        # Get the uid combinations
        uid_pnes     = bandpower_pnes.groupby(['uid']).indices
        uid_epilepsy = bandpower_epilepsy.groupby(['uid']).indices

        # Make an output objects
        cols = []
        for ikey in uid_epilepsy.keys():
            cols.append(ikey)
        for ikey in uid_pnes.keys():
            cols.append(ikey)
        for icol in bandpower_fcols:
            cols.append(f"distcomp_{icol}")
        for icol in timeseries_fcols:
            cols.append(f"distcomp_{icol}")

        ntrial  = 100000
        results = []
        for itrial in tqdm(range(ntrial), total=ntrial, desc="Bootstrap analysis."):

            # Grab a random subset
            epilepsy_inds = []
            pnes_inds     = []
            for ikey in uid_pnes.keys():
                pnes_inds.append(np.random.choice(uid_pnes[ikey]))
            for ikey in uid_epilepsy.keys():
                epilepsy_inds.append(np.random.choice(uid_epilepsy[ikey]))
            
            # Grab a dataslice
            bandpower_pnes_slice      = bandpower_pnes.iloc[pnes_inds]
            bandpower_epilepsy_slice  = bandpower_epilepsy.iloc[epilepsy_inds]
            timeseries_pnes_slice     = timeseries_pnes.iloc[pnes_inds]
            timeseries_epilepsy_slice = timeseries_epilepsy.iloc[epilepsy_inds]

            # Make the value array to send to the summary
            output  = []
            for ival in epilepsy_inds:output.append(ival)
            for ival in pnes_inds:output.append(ival)
            for icol in bandpower_fcols:
                val1 = list(bandpower_epilepsy_slice[icol].values)
                val2 = list(bandpower_pnes_slice[icol].values)
                output.append(ttest_ind(val1,val2)[1])
            for icol in timeseries_fcols:
                val1 = list(timeseries_epilepsy_slice[icol].values)
                val2 = list(timeseries_pnes_slice[icol].values)
                output.append(ttest_ind(val1,val2)[1])

            results.append(output)

        output  = PD.DataFrame(np.array(results),columns=cols)
        output.to_csv(f"{outdir}bootstrap_stats.csv",index=False)

    def calculate_anova(self):
                
        # Make a list of vectors to not test
        blacklist = ['file', 't_start', 't_end', 't_window', 'uid', 'target']
        features  = np.setdiff1d(self.data.columns,blacklist)
        target    = self.data['target'].values

        # Make the ANOVA dataframe
        rows = []
        for icol in features:
            idata                = self.data[icol].values.reshape((-1,1))
            f_statistic, p_value = f_classif(idata, target)
            rows.append([icol,f_statistic[0],p_value[0]])
        self.anova_df = PD.DataFrame(rows,columns=['feature','f-stat','p-val'])
        self.anova_df = self.anova_df.set_index('feature')

    def paired_whisker_plotter(self,iDF,exog_col,plotcols,outdir,ydown=.25,yup=0.5,bonustitle=None):

        # Grab the unique values in the exogenous column
        exog_vals = np.unique(iDF[exog_col].values)
        all_exog  = iDF[exog_col].values
        
        # Loop over the columns to plot
        for icol in plotcols:

            # Get the values shared between patients for the exogenous column
            DF_grp      = iDF.groupby(['uid',exog_col],as_index=False)[icol].median()
            DF_0        = DF_grp.loc[DF_grp[exog_col]==exog_vals[0]]
            DF_1        = DF_grp.loc[DF_grp[exog_col]==exog_vals[1]]
            shared_uids = np.intersect1d(DF_0.uid.values,DF_1.uid.values)
            vals_0      = DF_0.loc[DF_grp['uid'].isin(shared_uids)][icol]
            vals_1      = DF_1.loc[DF_grp['uid'].isin(shared_uids)][icol]

            # get the paired t-test values
            t_score,pval = ttest_rel(vals_0,vals_1)

            # Make a title string with stats
            title_str = f"{icol}\nP-value between {exog_col} {exog_vals[0]} and {exog_vals[1]}:{pval:.2e}"
            if bonustitle != None:
                title_str += f" ({bonustitle})"

            # Make a plotting dataframe
            all_vals  = np.concatenate((vals_0,vals_1))
            all_exog  = [exog_vals[0] for ii in range(vals_0.size)] 
            all_exog.extend([exog_vals[1] for ii in range(vals_1.size)])
            pDF           = PD.DataFrame(all_vals,columns=[icol])
            pDF[exog_col] = all_exog
            pDF['x']      = icol

            # Make a slightly better plotting scale
            all_vals   = pDF[icol].values
            zvals      = np.fabs(zscore(all_vals))
            yrange_raw = np.ceil(np.fabs(np.interp([1],zvals,all_vals)[0]))
            yrange     = [-yrange_raw+ydown,yrange_raw+yup]

            # Make the plot
            print(f"Plotting {icol}")
            fig = PLT.figure(dpi=100,figsize=(6.,6.))
            ax  = fig.add_subplot(111)
            sns.boxplot(data=pDF,x='x',y=icol,hue=exog_col, boxprops={'alpha': 0.4}, ax=ax)
            sns.stripplot(data=pDF, x="x", y=icol, hue=exog_col, dodge=True, ax=ax)
            ax.set_xlabel(f"Feature",fontsize=13)
            ax.set_ylabel(f"{icol}",fontsize=13)
            ax.set_title(title_str,fontsize=13)
            try:
                ax.set_ylim(yrange)
            except:
                pass
            PLT.savefig(f"{outdir}{icol}.png")
            PLT.close("all")

    def plot_paired_whisker_pnes_vs_epilepsy(self,outdir):

        # Make the mixed version of target
        smap = {1:'sleep',2:'wake'} 
        iDF  = self.data.loc[self.data.sleep_state.isin([1])]
        self.paired_whisker_plotter(iDF,'target',self.model_block['bandpower'],outdir+'SLEEP/',ydown=0.85,yup=-.45,bonustitle="sleep")
        self.paired_whisker_plotter(iDF,'target',self.model_block['timeseries'],outdir+'SLEEP/',ydown=0.5,yup=-.25,bonustitle="sleep")
        iDF  = self.data.loc[self.data.sleep_state.isin([2])]
        self.paired_whisker_plotter(iDF,'target',self.model_block['bandpower'],outdir+'WAKE/',ydown=0.85,yup=-.45,bonustitle="wake")
        self.paired_whisker_plotter(iDF,'target',self.model_block['timeseries'],outdir+'WAKE/',ydown=0.5,yup=-.25,bonustitle="wake")

    def plot_paired_whisker_sleep_vs_wake(self,outdir):

        # Create a more human readable label
        smap = {1:'sleep',2:'wake'} 

        # Plot differences in bandpower columns
        iDF  = self.data.loc[self.data.sleep_state.isin([1,2])]
        iDF['sleep_state'] = iDF['sleep_state'].apply(lambda x:smap[x])
        self.paired_whisker_plotter(iDF,'sleep_state',self.model_block['bandpower'],outdir)
        self.paired_whisker_plotter(iDF,'sleep_state',self.model_block['timeseries'],outdir)

    def plot_paired_pdf(self,outdir):

        # Make a locally scoped data object
        pdf_data = self.data.copy()

        # Make a list of vectors to plot
        blacklist = ['file', 't_start', 't_end', 't_window', 'uid']
        pdf_data  = pdf_data.drop(blacklist,axis=1)
        features  = np.setdiff1d(pdf_data.columns,['target'])

        # Make the plot of each columns
        for icol in features:

            # Make a z-score outlier rejection
            pdf_data['z'] = zscore(pdf_data[icol].values)
            mask          = np.fabs(pdf_data['z'].values)<5
            iDF           = pdf_data.iloc[mask]

            # Get the PNES and Epilelpsy values
            pnes_mask     = (iDF['target'].values=='pnes')
            epilepsy_mask = (iDF['target'].values=='epilepsy')

            # Get the values
            pnes_vals     = iDF[icol].values[pnes_mask]
            epilepsy_vals = iDF[icol].values[epilepsy_mask]

            # Get the pdf
            x_min = min([np.amin(pnes_vals),np.amin(epilepsy_vals)])
            x_max = max([np.amax(pnes_vals),np.amax(epilepsy_vals)])
            xbin  = np.linspace(x_min,x_max,200)
            pnes_counts, pnes_edges = np.histogram(pnes_vals,bins=xbin)
            centers  = (pnes_edges[:-1]+pnes_edges[1:])
            pnes_pdf = pnes_counts/pnes_counts.sum()
            epilepsy_counts, epilepsy_edges = np.histogram(epilepsy_vals,bins=xbin)
            epilepsy_pdf = epilepsy_counts/epilepsy_counts.sum()

            # Make the plots
            fig  = PLT.figure(dpi=100,figsize=(6.,6.))
            ax   = fig.add_subplot(111)
            pobj = ax.step(centers,pnes_pdf,color='k')
            eobj = ax.step(centers,epilepsy_pdf,color='r')
            ax.set_xlabel(f"Feature (x)",fontsize=16)
            ax.set_ylabel(f"P(x)",fontsize=16)
            ax.set_title(f"{icol}",fontsize=16)
            PLT.savefig(f"{outdir}{icol}.png")
            PLT.close("all")

    def plot_vectors(self,outdir):

        # Grab the data in the same we will in MLP to make sure it matches
        X_train_bandpower   = self.mlp_object[0]
        X_test_bandpower    = self.mlp_object[1]
        X_train_timeseries  = self.mlp_object[2]
        X_test_timeseries   = self.mlp_object[3]
        X_train_categorical = self.mlp_object[4]
        X_test_categorical  = self.mlp_object[5]
        Y_train             = self.mlp_object[6]
        Y_test              = self.mlp_object[7]
        model_block         = self.mlp_object[8]

        for icol in range(X_train_bandpower.shape[1]):
            vals    = X_train_bandpower[:,icol]
            colname = model_block['bandpower'][icol] 
            
            fig = PLT.figure(dpi=100,figsize=(8.,6.))
            ax  = fig.add_subplot(111)
            ax.hist(vals,bins=200,histtype='step')
            ax.set_title(f"{colname}", fontsize=14)
            PLT.savefig(f"{outdir}bandpower_train_{icol}")
            PLT.close("all")

        for icol in range(X_test_bandpower.shape[1]):
            vals    = X_test_bandpower[:,icol]
            colname = model_block['bandpower'][icol]

            fig = PLT.figure(dpi=100,figsize=(8.,6.))
            ax  = fig.add_subplot(111)
            ax.hist(vals,bins=200,histtype='step')
            ax.set_title(f"{colname}", fontsize=14)
            PLT.savefig(f"{outdir}bandpower_test_{icol}")
            PLT.close("all")

        for icol in range(X_train_timeseries.shape[1]):
            vals    = X_train_timeseries[:,icol]
            colname = model_block['timeseries'][icol]
            
            fig = PLT.figure(dpi=100,figsize=(8.,6.))
            ax  = fig.add_subplot(111)
            ax.hist(vals,bins=200,histtype='step')
            ax.set_title(f"{colname}", fontsize=14)
            PLT.savefig(f"{outdir}timeseries_train_{icol}")
            PLT.close("all")

        for icol in range(X_test_timeseries.shape[1]):
            vals    = X_test_timeseries[:,icol]
            colname = model_block['timeseries'][icol]
            
            fig = PLT.figure(dpi=100,figsize=(8.,6.))
            ax  = fig.add_subplot(111)
            ax.hist(vals,bins=200,histtype='step')
            ax.set_title(f"{colname}", fontsize=14)
            PLT.savefig(f"{outdir}timeseries_test_{icol}")
            PLT.close("all")

    def quantile_features(self,outdir):

        # get the sleep data
        sleep_DF = self.data.loc[self.data.sleep_state==1]

        # Get the feature columns
        localcols = ['file','uid','t_start','target']
        outcols   = []
        for icol in sleep_DF.columns:
            if 'C03' in icol or 'C04' in icol:
                if 'welch' in icol:
                    if '1.00' in icol:
                        outcols.append(icol)
                else:
                    if 'quantile' not in icol:
                        outcols.append(icol)
        allcols = localcols+outcols

        # Make an output dataframe
        outDF = sleep_DF[allcols].copy()

        # get the quantiles
        outcols2 = localcols.copy()
        qx       = np.linspace(0,1,1000)
        for icol in outcols:

            newcol = f"QUANTILE_{icol}"
            outcols2.append(icol)
            outcols2.append(newcol)

            # Get the quantile
            vals        = outDF[icol].values
            qy          = np.quantile(vals,q=qx)
            outDF[newcol] = np.interp(vals,qy,qx)
        
        # Save the results
        outDF = outDF[outcols2].sort_values(by=localcols)
        outDF.to_csv(f"{outdir}feature_quantiles.csv",index=False)

    def linear_seperability_search(self,outdir):

        # Get the feature columns
        blacklist = ['t_start', 't_end', 't_window', 'sleep_state']
        localcols = ['file','uid','target']
        slicecol  = np.setdiff1d(self.data.columns,blacklist)
        fcols     = np.setdiff1d(slicecol,localcols)

        # Make breaks for different sleep states
        sleep_DF = self.data.loc[self.data.sleep_state==1][slicecol]
        wake_DF  = self.data.loc[self.data.sleep_state==2][slicecol]

        # Make PNES and Epilepsy cohorts
        sleep_pnes = sleep_DF.loc[sleep_DF.target=='pnes']
        sleep_epi  = sleep_DF.loc[sleep_DF.target=='epilepsy']

        # Add in the counts to the target labels
        n_pnes = sleep_DF.loc[sleep_DF['target']=='pnes']['uid'].nunique()
        n_epi  = sleep_DF.loc[sleep_DF['target']=='epilepsy']['uid'].nunique()
        tvals  = sleep_DF['target'].values
        tvals[tvals=='pnes']     = f"pnes(N_sub={n_pnes})"
        tvals[tvals=='epilepsy'] = f"epilepsy(N_sub={n_epi})"
        sleep_DF['target'] = tvals

        # Loop over the columns and get cdf
        for icol in tqdm(fcols, total=fcols.size):
            
            # Grab the raw values
            pnes_vals = sleep_pnes[icol].values.flatten()
            epi_vals  = sleep_epi[icol].values.flatten()

            # get the p-value
            tval,pval = ttest_ind(pnes_vals,epi_vals)

            # Make the binning
            allmin  = np.amin([np.amin(pnes_vals),np.amin(epi_vals)])
            allmax  = np.amax([np.amax(pnes_vals),np.amax(epi_vals)])
            bins    = np.logspace(allmin,allmax,1000)
            centers = 0.5*(bins[:-1]+bins[1:])

            # get the y range
            rawvals = np.concatenate((pnes_vals,epi_vals))
            ymin    = np.quantile(rawvals,q=0.1)
            ymax    = np.quantile(rawvals,q=0.9)

            # Get the new dataslice for whisker plots
            icols    = localcols.copy()
            icols.append(icol)
            iDF      = sleep_DF[icols].copy()

            # get the group values
            jDF            = iDF.groupby(['uid'],as_index=False)[icol].median()
            target_groups  = iDF.groupby(['uid'],as_index=False)['target'].first()
            merged_DF      = PD.merge(jDF, target_groups, on='uid', how='inner')
            merged_DF['x'] = icol

            # Make the plots
            titlestr = f"{icol}\nP-value:{pval:.3e}"
            fig = PLT.figure(dpi=100,figsize=(8.,6.))
            ax  = fig.add_subplot(111)
            sns.boxplot(data=merged_DF,x='x',y=icol,hue='target', ax=ax, boxprops={'alpha': 0.4})
            sns.stripplot(data=merged_DF, x="x", y=icol, hue='target', dodge=True, ax=ax)
            ax.set_title(titlestr,fontsize=14)
            PLT.savefig(f"{outdir}{icol}.png")
            PLT.close("all")

