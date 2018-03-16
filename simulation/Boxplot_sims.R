#! /usr/bin/env Rscript

q=read.table('lin.out')
a=read.table('epi.out')
q=subset(q,q$V4>0.01 & q$V1!=5000)
par(mar=c(4.1,10.1,1.1,2.1),mfrow=c(1,2))
boxplot(q$V4~q$V3+q$V2+q$V1,las=2,horizontal=T,cex.axis=0.7,main='Additive arquitecture',col=5:8,ylim=c(0,0.4))
boxplot(a$V4~a$V3+a$V2+a$V1,las=2,horizontal=T,cex.axis=0.7,main='Epistatic arquitecture',col=5:8)
png('fig3.png',width=10,height=6)
png('fig3.png')
par(mar=c(4.1,10.1,1.1,2.1),mfrow=c(1,2))
boxplot(sqrt(q$V4)~q$V3+q$V2+q$V1,las=2,horizontal=T,cex.axis=0.7,main='Additive arquitecture',col=5:8)
boxplot(sqrt(a$V4)~a$V3+a$V2+a$V1,las=2,horizontal=T,cex.axis=0.7,main='Epistatic arquitecture',col=5:8)
dev.off()

#only median
boxplot(q$V4~q$V3+q$V2+q$V1,horizontal=T,add=T,outline = FALSE, boxlty = 0, whisklty = 0, staplelty = 0)
boxplot(count ~ spray, data = InsectSprays, outline = FALSE, boxlty = 0,
  whisklty = 0, staplelty = 0)