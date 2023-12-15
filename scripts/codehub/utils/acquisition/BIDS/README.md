# BIDS Data Generation

## EEG Data to BIDS

### iEEG.org

#### Sample Call
```
python utils/acquisition/EEG_BIDS.py --ieeg --username BJPrager --password ****** --bidsroot ../../user_data/BIDS/ --session preimplant --inputs_file ../../../scalp_deep-learning/user_data/targets.csv --annotations --multithread --ncpu 2
```
