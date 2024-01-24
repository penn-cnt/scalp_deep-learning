# BIDS Data Generation

BIDS data is a means of organizing and naming epilepsy datasets in a cross-platform multi-institute manner that removes confusion from differing naming schema.

## EEG Data to BIDS

When possible, we should aim to download data into a BIDS compliant format. The underlying data can still be accessed using direct invocation via your preferred language,  This code will create a bids dataset from
- Local files
- iEEG.org datasets.

The following sections detail how to 

### iEEG.org

#### Sample Call
```
python EEG_BIDS.py --ieeg --username BJPrager --password ****** --bidsroot ../../user_data/BIDS/ --session preimplant --inputs_file samples/targets.csv --annotations --multithread --ncpu 2
```

Where
- ieeg : Selects ieeg.org as the source data
- username : iEEG.org Username
- password : iEEG.ord Password
- bidsroot : Output BIDS top level directory to save to
- session : Session label for the dataset (i.e. preimplant, postsurg, etc.)
- inputs_file : File with iEEG.org filenames, unique patient identifiers, and any targets you wish to associate with each dataset
- annotations : Download by annotations
- multithread : Download and prepare BIDS data using multiple cpus
- ncpu : Number of cpus to use if multithread is selected

For more information and other options, please consult:
```
python EEG_BIDS.py --help
```
