1. 随着时间变化的HAMD值 查看不同药物的剂量的效果
	1. 回答的问题，什么时候会出现response

2. Do regression using task change multivariables
![[Pasted image 20240520134206.png]]

3. Is there any different for the task change after treatment?
	1. ==There were no changes from pretreatment to posttreatment==

5. Treatment response predict at different time line
	1. predict after T1/T2/T3... months 
	- Similarly, it would be important to evaluate whether different disruptions to intrinsic func- tional connectivity differentially predict outcomes for multiple different active treatments. Moreover, in this study treatment outcome was only assessed at the 8-week time point. Including multiple follow-ups before and after the 8-week time point would be important to determine whether predictive relationships change as a function of time during treatment.


6. Associations between Demographic/Clinical Characteristics and PCC-ACC/mPFC Connectivity
	- Thus, in supplementary linear regression analyses, we considered whether the role of intrinsic connectivity in predicting remission status may reflect differences in these predictor vari- ables at baseline or instead contribute independently to prediction beyond these other factors. At pretreatment baseline, we found that ACC/mPFC connectivity cluster was not associated with any of these variables (Supplemental Table S1).

Using SVM -> all task change feature

Only clinical data 

## Inner Cross-Validation Performance
| Classifier | Average bAcc | Average Sensitivity | Average Specificity | Average F1 Score |
|------------|-----------------|------------------|---------------------|---------------------|
| SVM | 0.4072 | 0.4036 | 0.4108 | 0.2267 |
| XGBoost | 0.5250 | 0.4318 | 0.6182 | 0.3032 |
| Naive Bayes | 0.4376 | 0.7052 | 0.1700 | 0.3037 |

## Outer Cross-Validation Performance
| Classifier | Average bAcc | Average Sensitivity | Average Specificity | Average F1 Score |
|------------|-----------------|------------------|---------------------|---------------------|
| SVM | 0.2929 | 0.2857 | 0.3000 | 0.1509 |
| XGBoost | 0.5043 | 0.4286 | 0.5800 | 0.2927 |
| Naive Bayes | 0.4843 | 0.9286 | 0.0400 | 0.3467 |

Clinical data + MDDR 