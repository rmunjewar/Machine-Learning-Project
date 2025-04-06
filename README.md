# Machine-Learning-Project

## Setting up the project

1. **Download the dataset**  
    Download the data from [KaggleHub](https://www.kaggle.com/datasets/ravi72munde/uber-lyft-cab-prices?resource=download) and save the dataset  into a folder called `dataset` in the root directory of this project. The dataset file should be named `cab_rides.csv`.
2. **Install dependencies**  
    Run the following command to install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
3. **Run the project**
    Execute the `main.py` file to pre-process the data (have not started on models yet):
4. **Error Handling**
If the dataset file is missing `dataset/cab_rides.csv`, the program will notify you to download the file manually. This is done because the files are too large for GitHub to upload correctly.

## Project Overview

This project analyzes the **cab_rides** dataset to predict the price of a given cab ride given the *distance*, *surge_multiplier*, and *cab_type*.

1. **Preprocessing**
    * Removes all rows with missing output feature *price*
    * Uses KNN Imputation to handle missing input input features *distance*, and *surge_multiplier*
    * Filles in missing data for the *cab_type* data with the most frequent value.

    KNN Imputation is used on the more important fields for determining the price of the cab trip, while a more simple method of selecting the mode works for the most frequent value to prevent the program from becoming cumbersome.