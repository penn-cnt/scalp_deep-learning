                self.iso_cols.append((icol,0.05))

    def data_rejection(self):

        # Alert user
        print("Running isolation forest to reject the most extreme time segments.")

        # Run an isolation forest across each feature in the training block
        self.iso_forests = {}
        for ipair in self.iso_cols:
            icol                   = ipair[0]
            contam_factor          = ipair[1]
            ISO                    = IsolationForest(contamination=contam_factor, random_state=42)
            self.iso_forests[icol] = ISO.fit(self.train_raw[icol].values.reshape((-1,1)))

        # Get the training mask
        train_2d_mask = np.zeros((self.train_raw.shape[0],len(self.iso_cols)))
        for idx,ipair in enumerate(self.iso_cols):
            icol                 = ipair[0]
            train_2d_mask[:,idx] = self.iso_forests[icol].predict(self.train_raw[icol].values.reshape((-1,1)))
        train_mask     = (train_2d_mask==1).all(axis=1)
        self.train_raw = self.train_raw.iloc[train_mask]
        
        # Get the testing mask
        test_2d_mask = np.zeros((self.test_raw.shape[0],len(self.iso_cols)))
        for idx,ipair in enumerate(self.iso_cols):
            icol                = ipair[0]
            test_2d_mask[:,idx] = self.iso_forests[icol].predict(self.test_raw[icol].values.reshape((-1,1)))
        test_mask     = (test_2d_mask==1).all(axis=1)
        self.test_raw = self.test_raw.iloc[test_mask]

        print(f"Isolation Forest reduced training size to {train_mask.sum()} from {train_mask.size} samples.")
        print(f"Isolation Forest reduced test size to {test_mask.sum()} from {test_mask.size} samples.")

    def scale_quantiles(self):

        # Get the channel names
        medcols     = []
        chan_lookup = {}
        for icol in self.indata.columns:
            if 'median' in icol:
                chan_lookup[icol.split('_')[0]] = self.indata[icol]
                medcols.append(icol)
        
        # Get scaled quantiles
        for ichan in chan_lookup.keys():
            for icol in self.indata.columns:
                if ichan in icol and 'quantile' in icol:
                    self.indata[icol] /= chan_lookup[ichan]

        # Drop the medians
        self.indata = self.indata.drop(medcols,axis=1)

    def float_binarize(self,Zcutoff=10,nbin=100):

        # Loop over the training vectors to get their bins
        self.train_binarize_bins = {}

        # Loop over the binarize columns
        for icol in self.bin_cols:

            # Derive the bins
            vals  = self.train_transformed[icol].values
            zvals = zscore(vals)
            loZ   = np.interp(-Zcutoff,zvals,vals)
            hiZ   = np.interp(Zcutoff,zvals,vals)
            bins  = np.linspace(loZ,hiZ)
            bins  = np.sort(np.append(bins,[-np.inf,np.inf]))

            # Transform the train data
            self.train_transformed[icol] = np.digitize(self.train_transformed[icol],bins)

            # Transform the test data
            self.test_transformed[icol] = np.digitize(self.test_transformed[icol],bins)

            # Save the bins for transforming the test data
            self.train_binarize_bins[icol] = bins

            vals = self.train_transformed[icol].values
            uvals,ucnts = np.unique(vals,return_counts=True)
            import pylab as PLT
            PLT.step(uvals,ucnts)
            PLT.show()
            exit()

        exit()





