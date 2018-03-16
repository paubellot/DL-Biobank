#!/bin/bash
source ~/Projects/biobank/biobank/bin/activate

# linear model
qaction=linear # additive genetic model
nloci=100 # num qtls in model (nloci=100, 1000, 10000)
k=1000    # num snps in model (k=10000, 20000, 50000 were evaluated)
for i in $(seq 1 10); do
    python simulation.py  --id $i$model$nloci --genmodel $qaction --loci $nloci --k $k --method lasso --recreation True
    python simulation.py  --id $i$model$nloci --genmodel $qaction --loci $nloci --k $k --method ridge 
    python simulation.py  --id $i$model$nloci --genmodel $qaction --loci $nloci --k $k --method mlp1 
    python simulation.py  --id $i$model$nloci --genmodel $qaction --loci $nloci --k $k --method mlp2 
done

# epistatic action
qaction=epistasia 
nloci=100 # num qtls in model (nloci=100, 10000)
k=1000    # num snps in model (k=10000, 20000, 50000 were evaluated)
for i in $(seq 1 10); do
    python simulation.py  --id $i$model$nloci --genmodel $qaction --loci $nloci --k $k --method lasso --recreation True
    python simulation.py  --id $i$model$nloci --genmodel $qaction --loci $nloci --k $k --method ridge 
    python simulation.py  --id $i$model$nloci --genmodel $qaction --loci $nloci --k $k --method mlp1 
    python simulation.py  --id $i$model$nloci --genmodel $qaction --loci $nloci --k $k --method mlp2 
done

