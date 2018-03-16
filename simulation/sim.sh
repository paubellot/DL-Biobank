#source biobank/bin/activate
for i in $(seq 1 2); do
    time python main.py --genmodel linear --loci 100 --k 10000 --method lasso --recreation True
    python main.py --genmodel linear --loci 100 --k 10000 --method ridge 
    python main.py --genmodel linear --loci 100 --k 10000 --method mlp1
    python main.py --genmodel linear --loci 100 --k 10000 --method mlp2 
done
