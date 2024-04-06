## 1. Install Environment (Should install conda first)

```
conda create -n tf python==3.9
conda activate tf
pip install scikit-learn==0.24.1 matplotlib==3.3.4 QtPy==1.9.0 jupyter==1.0.0 Keras==2.12.0 numpy==1.22.4 pandas==1.2.4 wandb==0.15.11 tensorflow-addons==0.20.0 tensorflow[and-cuda]==2.12.0 PyWavelets==1.4.1 PyWavelets==1.4.1 xgboost==2.0.1 scipy==1.9.1
```

---

## 2. Train the model 

```
nohup bash ./response_prediction.sh --model gnn_transformer --validation loocv --config pretreatment_response --msg test > /dev/null 2>&1 &
nohup bash ./response_prediction.sh --model gnn_transformer --validation loocv --config posttreatment_response --msg test > /dev/null 2>&1 &
nohup bash ./response_prediction.sh --model gnn_transformer --validation loocv --config pretreatment_remission --msg test > /dev/null 2>&1 &
nohup bash ./response_prediction.sh --model gnn_transformer --validation loocv --config posttreatment_remission --msg test > /dev/null 2>&1 &


./response_prediction.sh --model fusion_xgboost --validation loocv --config fusion_pretreatment_response --msg test
```

---

## 3. To read the result

- Validation method: loocv
```
python scripts/plot/DL/read_LOO_nestedCV_gnntr.py --model gnn_transformer --max 4 --dataset pretreatment_response 

python scripts/plot/DL/read_SCVHO.py --model gnn_transformer --dataset posttreatment_response --max 5


python scripts/plot/DL/read_LOO_nestedCV_gnntr.py --model gnn_transformer --max 1 --dataset posttreatment_remission --value_add_to_sensitivity_value 0.1
python scripts/plot/DL/read_LOO_nestedCV_gnntr.py --model fusion_xgboost --max 2 --dataset pretreatment_response

```

- Validation method: Stratified CV with hold out
```
python scripts/plot/DL/read_SCVHO.py --model gnn_transformer --max 4
```

note: --max is the maximum iteration for each training fold that can be achieved.


---

## 4. SHAP explaining the model

---

### Explaination of different files
- train.py 
    - Description: use this to train the deep learning model 

- config.py 
    - Description: this file consists data path you need to modify



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



