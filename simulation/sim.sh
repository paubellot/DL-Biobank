#!/bin/bash

## linear model nloci=1000
./sim1.sh
## model with epistasia nloci=1000
./sim2.sh
## linear model nloci=5000
./sim2.sh
## new gamma
./sim4.sh

## Plot in R
grep Names nohup.lin.*|grep -v nan| awk '{print $5,$4,$6,$12}'|sed s/=/' '/g| sed s/,//g| sed s/\'/' '/g | awk '{print $2,$4,$6,$7}' > lin.out
grep Names nohup.epi.*|grep -v nan|awk '{print $5,$4,$6,$12}'|sed s/=/' '/g| sed s/,//g| sed s/\'/' '/g | awk '{print $2,$4,$6,$7}' > epi.out
Rscript a.R
