PROCESS_ID=$(ps aux | grep "./run_automl_v0.sh" | grep -v grep | awk '{print $2}')
if [ ! -z "$PROCESS_ID" ]; then
  echo "Actual process id: $PROCESS_ID" 
  kill $PROCESS_ID
fi

PROCESS_ID=$(ps aux | grep "./run_automl.sh" | grep -v grep | awk '{print $2}')
if [ ! -z "$PROCESS_ID" ]; then
  echo "Actual process id: $PROCESS_ID" 
  kill $PROCESS_ID
fi
PROCESS_ID=$(ps aux | grep "./run_python_train.sh" | grep -v grep | awk '{print $2}')
if [ ! -z "$PROCESS_ID" ]; then
  echo "Actual process id: $PROCESS_ID" 
  kill $PROCESS_ID
fi
PROCESS_ID=$(ps aux | grep "./response_prediction.sh" | grep -v grep | awk '{print $2}')
if [ ! -z "$PROCESS_ID" ]; then
  echo "Actual process id: $PROCESS_ID" 
  kill $PROCESS_ID
fi

PROCESS_ID_PYTHON=$(ps aux | grep "LOOCV_train.py" | grep -v grep | awk '{print $2}')
if [ ! -z "$PROCESS_ID_PYTHON" ]; then 
    echo "Python Process id": $PROCESS_ID_PYTHON 
    kill $PROCESS_ID_PYTHON
fi

PROCESS_ID_PYTHON=$(ps aux | grep "LOO_nested_CV_train.py" | grep -v grep | awk '{print $2}')
if [ ! -z "$PROCESS_ID_PYTHON" ]; then 
    echo "Python Process id": $PROCESS_ID_PYTHON 
    kill $PROCESS_ID_PYTHON
fi


PROCESS_ID_PYTHON=$(ps aux | grep "train_transformer.py" | grep -v grep | awk '{print $2}')
if [ ! -z "$PROCESS_ID_PYTHON" ]; then 
    echo "Python Process id": $PROCESS_ID_PYTHON 
    kill $PROCESS_ID_PYTHON
fi

PROCESS_ID_PYTHON=$(ps aux | grep "StratifiedKFold_holdout_train.py" | grep -v grep | awk '{print $2}')
if [ ! -z "$PROCESS_ID_PYTHON" ]; then 
    echo "Python Process id": $PROCESS_ID_PYTHON 
    kill $PROCESS_ID_PYTHON
fi