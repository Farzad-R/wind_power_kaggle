# Title

Wind Power Forecasting of Global Energy Forecasting Competition 2012 - Wind Forecasting

## Description

In decided to create this project in my spare time. Here I will try to prepare a full data analysis and prediction of wind power data. The main goal is to provide a proper way of addressing time series data rather than blindly unleashing various deep learning models. I will use jupyter notebook for data analysis and exploration and I will try to provide the necessary information and extra links along the way. The main pipeline will be in .py modules. You can find the data set [here](https://drive.google.com/file/d/1YRr9xB8X7QffXSYoaPw15yjNd-1Y-_we/view?usp=sharing).

## Project initial structure

Current structure of the project:
```
.
├── data                    # (added to .gitignore)
│   ├── transformed               # Clean data (auto-generate)
│   └── raw                 # The raw data
├── config                  # config files for the main pipeline
├── src                     # Contains the codes of the main pipeline
│   ├── utils               # Includes utils.py module
│   ├── EDA
│   ├── data_transform
│   └── data_clean
│
├── logs                    # Includes two levels of log files (TODO)
│   ├── debug.log               
│   └── info.log
│
├── requirements.txt        # Required packages for the project
├── README.md
├── .gitignore
├── setup.py				
└── test					# contains unit-test for different parts of the project (TODO)
```
### Pipeline Order:
	- download
	- transform: drop useles columns, make all column names lowercase, check column data types, set index if needed
	- clean: Any further cleaning needed beside the ones performed in transform (e.g: dropping extra rows, filtering met info based on data percentage,
	 filtering ieso and met based on min and maximum valid data etc.)
	- augment: adding nny necessary data to the transformed or cleaned datasets (e.g: wind_compass_direction, wind direction 360 degree altitude, etc.) 
	---> preprocess: preprocessing tasks(e.g: feature extraction, interpolation, merge, etc.)			 <---
	│																										 │
	---> evaluate: evaluating the output of preprocessing step (e.g: feature importance selection, etc.) <---
	- train: training using the outputs of preprocessing ()
	- ... TODO

### Command Order:
TODO
### Data and Competition Page:
[Kaggle competition link](https://drive.google.com/file/d/1YRr9xB8X7QffXSYoaPw15yjNd-1Y-_we/view?usp=sharing)

### Dependencies
TODO
## Authors

[Farzad Roozitalab](https://www.linkedin.com/in/farzad-roozitalab-173066152/)