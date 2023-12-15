# BIDS Data Generation

## EEG Data to BIDS

### iEEG.org

#### Sample Call
```
python utils/acquisition/EEG_BIDS.py --ieeg --username BJPrager --password AuraHimitsu42 --bidsroot ../../user_data/multiscratch --session preimplant --inputs_file ../../../scalp_deep-learning/user_data/test_targets.csv --annotations --multithread --ncpu 2
```
