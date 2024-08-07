def test_input_data(args,files,start_times,end_times):
    
    # Get the pathing to the excluded data
    if args.exclude == None:
        exclude_path = args.outdir+"excluded.txt"
    else:
        exclude_path = args.exclude

    # Get the files to use and which to save
    good_index = []
    bad_index  = []
    if os.path.exists(exclude_path):
        excluded_files = PD.read_csv(exclude_path)['file'].values
        for idx,ifile in enumerate(files):
            if ifile not in excluded_files:
                good_index.append(idx)
    else:
        # Confirm that data can be read in properly
        excluded_files = []
        for idx,ifile in enumerate(files):
            DLT  = data_loader_test()
            flag = DLT.edf_test(ifile)
            if flag[0]:
                good_index.append(idx)
            else:
                excluded_files.append([ifile,flag[1]])
        excluded_df = PD.DataFrame(excluded_files,columns=['file','error'])
        if not args.debug:
            excluded_df.to_csv(exclude_path,index=False)
    return files[good_index],start_times[good_index],end_times[good_index]


        # Consolidate the metadata for failed chunks and successful chunks, and squeeze the successful object to match output list
        ### There is an ocassional bug in how keys get handled. For now, blocking this code out and recommending people do not use data viability.
        """
        metadata_copy = self.metadata.copy()
        bad_metadata_keys = np.setdiff1d(list(self.metadata.keys()),self.output_meta)
        if bad_metadata_keys.size > 0:
            bad_metadata = self.metadata[bad_metadata_keys]
        else:
            bad_metadata = {}
        self.metadata = {}
        for idx,ikey in enumerate(self.output_meta):
            self.metadata[idx] = metadata_copy.pop(ikey)
        """
    



        def load_ssh(self,filetype):

        if filetype.lower() == 'ssh_edf':
            # Make a name for the temporary file
            file_path = ''.join(random.choices(string.ascii_letters + string.digits, k=8))+".edf"

            # Make a temporary file to be deleted at the end of this process
            data_stream.ssh_copy(self,self.infile,file_path,self.ssh_host,self.ssh_username)

            success_flag = True
            try:
                # Test the header
                read_edf_header(file_path)

                # If successful read the actual data
                raw = read_raw_edf(file_path,verbose=False)
            except:
                success_flag = False

            if success_flag:
                # Munge the data into the final objects
                self.indata   = raw.get_data().T
                self.channels = raw.ch_names
                self.sfreq    = raw.info.get('sfreq')

                # Keep a static copy of the channels so we can just reference this when using the same input data
                self.channel_metadata = self.channels.copy()                

            os.remove(file_path)
            return success_flag

    def load_iEEG(self,username,password,dataset_name):

        # Load current data into memory
        if self.infile != self.oldfile:
            with Session(username,password) as session:
                dataset     = session.open_dataset(dataset_name)
                channels    = dataset.ch_labels
                self.indata = dataset.get_data(0,np.inf,range(len(channels)))
            session.close_dataset(dataset_name)
        
        # Save the channel names to metadata
        self.channels = channels
        metadata_handler.set_channels(self,self.chanels)
        
        # Calculate the sample frequencies
        sample_frequency = [dataset.get_time_series_details(ichannel).sample_rate for ichannel in self.channels]
        metadata_handler.set_sampling_frequency(self,sample_frequency)
