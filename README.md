

# Getting Started

## First - write a task file 

In `tasks/file.sh`, you should write the following parameters:

1. __model_names__: Specify which model you will use (can be multiples).
2. __config_names__:  Specify which config you will use (can be multiples).
3. __run_itr__: Specify which config you will use (only support one now).
4. __seeds__: This will change the input data shuffling and change train, val, test data, and also data augmentation. 
5. __launcher_name__: Launching python file to start training and predicting.
6. __db_file__: SQlite3 path that will store the result.


To start a task, running this:
```
./run.sh
```



# Design of framework

![Framework and SQLite Database Design](./utils/imgs/SQLite%20databased%20design%20for%20MDD%20Classification%20V2.png)

## 1. Install Environment (Should install conda first)

```
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


This repo used SQLite to save and read results.

## Initial SQLite 

```
sqlite3 ./results/experiment_results.db < ./scripts/SQLite/sql/init_db.sql
```

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