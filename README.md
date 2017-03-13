# soybean

## How to run
1. Make sure you have python 2.7 and pandas installed
2. In terminal, run `$> ./run.sh`
3. (Optional) To run soy.ipynb, install jupyter and run `$> jupyter notebook soy.ipynb`

## How to generate training data
run
```bash
python prepare.py soy-yield.csv geo-info.csv gene-info.csv
```
replace those csv files with actual path
