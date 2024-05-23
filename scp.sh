scp -r ./allData $s5:~/Documents/fnirs/treatment_response/fnirs-depression-deeplearning/


scp -r $s5:~/Documents/fnirs/treatment_response/fnirs-depression-deeplearning/results/ML_results/pre_post_treatment_hamd_reduction_50 ./results/ML_results/


rsync -avz --progress $s5:~/Documents/fnirs/treatment_response/fnirs-depression-deeplearning/results/ML_results/pre_post_treatment_hamd_reduction_50 ./results/ML_results/

rsync -avz --progress $s5:~/Documents/fnirs/treatment_response/fnirs-depression-deeplearning/results/ML_results/pre_post_treatment_hamd_reduction_50 ./results/ML_results/


rsync -avz --progress ./allData $s5:~/Documents/fnirs/treatment_response/fnirs-depression-deeplearning
 $s5:~/Documents/fnirs/treatment_response/fnirs-depression-deeplearning/results/ML_results/pre_post_treatment_hamd_reduction_50 ./results/ML_results/



scp jy@100.122.67.5:/home/jy/Documents/fnirs/treatment_response/fnirs-depression-deeplearning/allData/fNIRSxMDDData_Demographics_Clinical.xlsx /Users/shanxiafeng/Desktop/csc/


rsync -avz --progress $s5:~/Documents/fnirs/treatment_response/fnirs-depression-deeplearning/allData ./allData 
rsync