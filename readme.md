# Quick notes for Sarah

## Installation
- Recommend you get started by cloning this on your laptop; getting some example neural data to start with (try 2-3 days worth of data)
- Git clone, cd to the directory
- `pip install -e .`

Repo is still in progress. Start by
- In `nquality/raw_data_template.py`, try to initialize the `data= SessionNeuralData(*)` object using an example session data 
- Then, in `nquality/quality_within_session.py`, create a `x = Session(data)` object using the `SessionNeuralData` object you just created. 
- Try to access the `x.ds_quality` property. This will crunch all the numbers 
