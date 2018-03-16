#!/bin/bash
source ~/Projects/biobank/biobank/bin/activate
model=epistasia
nloci=1000
for i in $(seq 1 10); do
    k=10000
    time python main.py --id $i$model$nloci --genmodel $model --loci $nloci --k $k --method lasso --recreation True
    for k in 20000 50000; do
        python main.py --id $i$model$nloci --genmodel $model --loci $nloci --k $k --method lasso
    done
    for k in 10000 20000 50000; do
        for method in ridge mlp1 mlp2; do
            python main.py  --id $i$model$nloci --genmodel $model --loci $nloci --k $k --method $method
        done
    done
done