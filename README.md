


### Explaination of different files
- train_transformer.py 
    - Description: use this to train the deep learning model 

- config.py 
    - Description: this file consists data path you need to modify



### For simple running the code and training 

- simply use this bash file to train the file
    > `scripts/bash/response_prediction.sh`

- use this bash file to terminate the file
    > `scripts/bash/clear.sh`




### Compare and run different tradition machine learning models 
- `./scripts/ML/train_5_models.py`
    - "Random Forest": RandomForestClassifier(),
    - "SVM": SVC(),
    - "KNN": KNeighborsClassifier(),  # 注意：KNN通常没有random_state参数
    - "Decision Tree": DecisionTreeClassifier()

### To validate and reproduce the result
- `./scripts/ML/validate_model.py`
