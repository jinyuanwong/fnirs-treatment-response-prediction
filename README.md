

# Getting Started

## Installing Enviroment 

```sh
conda create -n tf python==3.9
conda init
conda activate tf

pip install scikit-learn==0.24.1 matplotlib==3.3.4 QtPy==1.9.0 jupyter==1.0.0
pip install Keras==2.12.0 numpy==1.22.4 pandas==1.2.4 wandb==0.15.11
pip install tensorflow-addons==0.20.0 tensorflow[and-cuda]==2.12.0
pip install PyWavelets==1.4.1 xgboost==2.0.1 scipy==1.9.1
pip install statsmodels==0.14.1 hyperopt==0.2.7 shap==0.43.0
pip install chord==6.0.1 pingouin==0.5.4
pip install einops==0.8.0
pip install datasets==2.19.2
pip install dataclasses==0.6
```


## Initial SQLite 

This repo used SQLite to save and read results. As using a Database to store result should be more convenient as we and move the result and read the results easily using a Database.

### The design of the database

![Framework and SQLite Database Design](./utils/imgs/SQLite%20databased%20design%20for%20MDD%20Classification%20V2x.png)


### To initilise the database

```
sqlite3 ./results/experiment_results.db < ./scripts/SQLite/sql/init_db_v2.sql
```

## First - Write a Task File

In `tasks/file.sh`, you should define the following parameters:

1. **model_names**: Specify the models you will use (can be multiple).
2. **config_names**: Specify the configurations you will use (can be multiple).
3. **run_itr**: Specify the iteration for the run (only supports one currently).
4. **seeds**: Define the seeds to change input data shuffling, and affect train, validation, test data, and data augmentation.
5. **launcher_name**: The Python file used to start training and predicting.
6. **db_file**: The path to the SQLite3 database that will store the results.

## Running a Task

To start a task, you should modify the task_path in the `./run.sh` and type the following command:
```sh
./run.sh
```



# Design of framework









- Referernce of SQLite 
    - [Who needs MLflow when you have SQLite?](https://ploomber.io/blog/experiment-tracking/)

---

## 2. Train the model 

```
nohup ./run.sh &
```

---

## 3. To read the result


---

## 4. SHAP explaining the model



### For simple running the code and training 

- simply use this bash file to train the file
    > `scripts/bash/response_prediction.sh`

- use this bash file to terminate the file


### Compare and run different tradition machine learning models 
- `./scripts/ML/train_5_models.py`
    - "Random Forest": RandomForestClassifier(),
    - "SVM": SVC(),
    - "KNN": KNeighborsClassifier(),  # 注意：KNN通常没有random_state参数
    - "Decision Tree": DecisionTreeClassifier()

### To validate and reproduce the result
- `./scripts/ML/validate_model.py`



# File function
- DL_main.ipynb 
 - plot all and average metircs of GNN and save into a csv file call metrics.txt under results/prognosis_mix_hb