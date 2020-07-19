.libPaths("C:/Utenti/giuseppe.maulucci/OneDrive - Universit? Cattolica del Sacro Cuore/Rdir/library")
library(ggplot2)
library(readxl)
library(dplyr)
library(grid)
library(gridExtra)
library(gvlma)
library(car)
library(plotly)
library(rgl)
library(xfun)
library(corrplot)
library(compute.es)
library(effects)
library(multcomp)
library(pastecs)

DatabaseCompleto_Aprile2020 <- read_excel("C:/Users/giuseppe.maulucci/OneDrive - Universit? Cattolica del Sacro Cuore/Rdir/DatabaseCompleto_Aprile2020.xlsx")
View(DatabaseCompleto_Aprile2020)

data_filterx <-filter(Database_SEPT) 
View(data_filterx)
data_filter<-Database_SEPT[-c(5:16,141:158 ), ]

data_filterT2DM<-filter(data_filter, GROUP=="T2DM")
data_filterCTRL<-filter(data_filter, GROUP=="CTRL")
data_filterMCV<-filter(data_filter, GROUP=="T2DM+MCV")

horiz<- data_filter$GROUP
variable <- data_filter$GP_15



#box plot gruppi+anova GP_15
bxGP<-data_filter%>% ggplot(aes(x=horiz, y=variable))+ geom_boxplot()+  stat_summary(fun.y=mean, colour="blue", geom="point", shape=2, size=3)+theme_classic(base_size=8)+labs(x="Group",y="GP")#+geom_point(position = "jitter")#1+ scale_y_continuous(limits = c(0.4, 0.9))
res.aov <- aov(variable ~ horiz, data = data_filter)
summary(res.aov)
TukeyHSD(res.aov)
#summary(glht(res.aov, linfct = mcp(group = "Tukey")))
bxGP


#MEDIE 
app1<- data_filterCTRL$GP_15
app2<-data_filterT2DM$GP_15
app3 <-data_filterMCV$GP_15

a<-mean(app1, na.rm=TRUE )
d<-sd(app1, na.rm=TRUE)
b<-mean(app2, na.rm=TRUE )
e<-sd(app2, na.rm=TRUE)
c<-mean(app3, na.rm=TRUE )
f<-sd(app3, na.rm=TRUE)
table_mean <- matrix(c(a,b,c,d,e,f), ncol=2)
colnames(table_mean)<- c("mean", "sd")
rownames(table_mean) <- c("CTRL","T2DM", "T2DM+MCV")
table_mean


#box plot gruppi+anova GPdev
#bxGPdev<-data_filter%>% ggplot(aes(x=horiz, y=variable2))+ geom_boxplot()+ stat_summary(fun.y=mean, colour="blue", geom="point", shape=2, size=3)+theme_classic(base_size=8)#+geom_point(position = "jitter")#1+ scale_y_continuous(limits = c(0.4, 0.9))
#res.aov <- aov(variable2 ~ horiz, data = data_filter)
#summary(res.aov)
#TukeyHSD(res.aov)

#box plot gruppi+anova HbA1C
bxHbA1C<-data_filter%>% ggplot(aes(x=GROUP, y=HB_GLICOSILATA))+ geom_boxplot()+  stat_summary(fun.y=mean, colour="blue", geom="point", shape=2, size=3)+theme_classic(base_size=8)#+ scale_y_continuous(limits = c(0.5, 0.7))
res.aov <- aov(HB_GLICOSILATA ~ horiz, data = data_filter)
summary(res.aov)
TukeyHSD(res.aov)

bxHbA1C
#MEDIE 
app1<- data_filterCTRL$HB_GLICOSILATA
app2<-data_filterT2DM$HB_GLICOSILATA
app3 <-data_filterMCV$HB_GLICOSILATA

a<-mean(app1, na.rm=TRUE )
d<-sd(app1, na.rm=TRUE)
b<-mean(app2, na.rm=TRUE )
e<-sd(app2, na.rm=TRUE)
c<-mean(app3, na.rm=TRUE )
f<-sd(app3, na.rm=TRUE)
table_mean <- matrix(c(a,b,c,d,e,f), ncol=2)
colnames(table_mean)<- c("mean", "sd")
rownames(table_mean) <- c("CTRL","T2DM", "T2DM+MCV")
table_mean


#box plot gruppi+anova LDL
bxLDL<-data_filter%>% ggplot(aes(x=GROUP, y=LDL))+ geom_boxplot()+  stat_summary(fun.y=mean, colour="blue", geom="point", shape=2, size=3)+theme_classic(base_size=8)#+ scale_y_continuous(limits = c(0.5, 0.7))
res.aov <- aov(LDL ~ GROUP, data = data_filter)
summary(res.aov)
TukeyHSD(res.aov)
bxLDL
#MEDIE 
app1<- data_filterCTRL$LDL
app2<-data_filterT2DM$LDL
app3 <-data_filterMCV$LDL

a<-mean(app1, na.rm=TRUE )
d<-sd(app1, na.rm=TRUE)
b<-mean(app2, na.rm=TRUE )
e<-sd(app2, na.rm=TRUE)
c<-mean(app3, na.rm=TRUE )
f<-sd(app3, na.rm=TRUE)
table_mean <- matrix(c(a,b,c,d,e,f), ncol=2)
colnames(table_mean)<- c("mean", "sd")
rownames(table_mean) <- c("CTRL","T2DM", "T2DM+MCV")
table_mean

#box plot gruppi+anova TCOL
bxTCOL<-data_filter%>% ggplot(aes(x=GROUP, y=COLESTEROLO_TOTALE))+ geom_boxplot()+  stat_summary(fun.y=mean, colour="blue", geom="point", shape=2, size=3)+theme_classic(base_size=8)+ scale_y_continuous(limits = c(0, 300))
res.aov <- aov(COLESTEROLO_TOTALE ~ GROUP, data = data_filter)
summary(res.aov)
TukeyHSD(res.aov)
bxTCOL
#MEDIE 
app1<- data_filterCTRL$COLESTEROLO_TOTALE
app2<-data_filterT2DM$COLESTEROLO_TOTALE
app3 <-data_filterMCV$COLESTEROLO_TOTALE

a<-mean(app1, na.rm=TRUE )
d<-sd(app1, na.rm=TRUE)
b<-mean(app2, na.rm=TRUE )
e<-sd(app2, na.rm=TRUE)
c<-mean(app3, na.rm=TRUE )
f<-sd(app3, na.rm=TRUE)
table_mean <- matrix(c(a,b,c,d,e,f), ncol=2)
colnames(table_mean)<- c("mean", "sd")
rownames(table_mean) <- c("CTRL","T2DM", "T2DM+MCV")
table_mean

#box plot gruppi+anova TG
bxTG<-data_filter%>% ggplot(aes(x=GROUP, y=TG))+ geom_boxplot()+  stat_summary(fun.y=mean, colour="blue", geom="point", shape=2, size=3)+theme_classic(base_size=8)+ scale_y_continuous(limits = c(0, 250))
res.aov <- aov(TG ~ GROUP, data = data_filter)
summary(res.aov)
TukeyHSD(res.aov)
bxTG

#box plot gruppi+anova HDL
bxHDL<-data_filter%>% ggplot(aes(x=GROUP, y=HDL))+ geom_boxplot()+  stat_summary(fun.y=mean, colour="blue", geom="point", shape=2, size=3)+theme_classic(base_size=12)#+ scale_y_continuous(limits = c(0.5, 0.7))
res.aov <- aov(HDL ~ GROUP, data = data_filter)
summary(res.aov)
TukeyHSD(res.aov)
bxHDL

#box plot gruppi+anova TGHDL
bxTGHDL<-data_filter%>% ggplot(aes(x=GROUP, y=TG/HDL))+ geom_boxplot()+  stat_summary(fun.y=mean, colour="blue", geom="point", shape=2, size=3)+theme_classic(base_size=8)+ scale_y_continuous(limits = c(0, 10))
res.aov <- aov(TG/HDL ~ GROUP, data = data_filter)
summary(res.aov)
TukeyHSD(res.aov)
bxTGHDL
#MEDIE 
app1<- data_filterCTRL$TG/HDL
app2<-data_filterT2DM$TG/HDL
app3 <-data_filterMCV$TG/HDL
a<-mean(app1, na.rm=TRUE )
d<-sd(app1, na.rm=TRUE)
b<-mean(app2, na.rm=TRUE )
e<-sd(app2, na.rm=TRUE)
c<-mean(app3, na.rm=TRUE )
f<-sd(app3, na.rm=TRUE)
table_mean <- matrix(c(a,b,c,d,e,f), ncol=2)
colnames(table_mean)<- c("mean", "sd")
rownames(table_mean) <- c("CTRL","T2DM", "T2DM+MCV")
table_mean



#box plot gruppi+anova TG
bxTG<-data_filter%>% ggplot(aes(x=GROUP, y=TG))+ geom_boxplot()+  stat_summary(fun.y=mean, colour="blue", geom="point", shape=2, size=3)+theme_classic(base_size=8)+ scale_y_continuous(limits = c(0, 250))
res.aov <- aov(TG ~ GROUP, data = data_filter)
summary(res.aov)
TukeyHSD(res.aov)
bxTG
#MEDIE 
app1<- data_filterCTRL$TG
app2<-data_filterT2DM$TG
app3 <-data_filterMCV$TG
a<-mean(app1, na.rm=TRUE )
d<-sd(app1, na.rm=TRUE)
b<-mean(app2, na.rm=TRUE )
e<-sd(app2, na.rm=TRUE)
c<-mean(app3, na.rm=TRUE )
f<-sd(app3, na.rm=TRUE)
table_mean <- matrix(c(a,b,c,d,e,f), ncol=2)
colnames(table_mean)<- c("mean", "sd")
rownames(table_mean) <- c("CTRL","T2DM", "T2DM+MCV")
table_mean






#box plot gruppi+anova HDL
bxHDL<-data_filter%>% ggplot(aes(x=GROUP, y=HDL))+ geom_boxplot()+  stat_summary(fun.y=mean, colour="blue", geom="point", shape=2, size=3)+theme_classic(base_size=12)#+ scale_y_continuous(limits = c(0.5, 0.7))
res.aov <- aov(HDL ~ GROUP, data = data_filter)
summary(res.aov)
TukeyHSD(res.aov)
bxHDL
#MEDIE 
app1<- data_filterCTRL$HDL
app2<-data_filterT2DM$HDL
app3 <-data_filterMCV$HDL
a<-mean(app1, na.rm=TRUE )
d<-sd(app1, na.rm=TRUE)
b<-mean(app2, na.rm=TRUE )
e<-sd(app2, na.rm=TRUE)
c<-mean(app3, na.rm=TRUE )
f<-sd(app3, na.rm=TRUE)
table_mean <- matrix(c(a,b,c,d,e,f), ncol=2)
colnames(table_mean)<- c("mean", "sd")
rownames(table_mean) <- c("CTRL","T2DM", "T2DM+MCV")
table_mean


#SOLO TABELLA

#box plot gruppi+anova AGE
bxAGE<-data_filter%>% ggplot(aes(x=GROUP, y=ETA))+ geom_boxplot()+  stat_summary(fun.y=mean, colour="blue", geom="point", shape=2, size=3)+theme_classic(base_size=8)#+ scale_y_continuous(limits = c(0.5, 0.7))
res.aov <- aov(ETA ~ GROUP, data = data_filter)
summary(res.aov)
TukeyHSD(res.aov)
#MEDIE 
app1<- data_filterCTRL$ETA
app2<-data_filterT2DM$ETA
app3 <-data_filterMCV$ETA
a<-mean(app1, na.rm=TRUE )
d<-sd(app1, na.rm=TRUE)
b<-mean(app2, na.rm=TRUE )
e<-sd(app2, na.rm=TRUE)
c<-mean(app3, na.rm=TRUE )
f<-sd(app3, na.rm=TRUE)
table_mean <- matrix(c(a,b,c,d,e,f), ncol=2)
colnames(table_mean)<- c("mean", "sd")
rownames(table_mean) <- c("CTRL","T2DM", "T2DM+MCV")
table_mean
#bxAGE

#box plot gruppi+anova BMI
bxBMI<-data_filter%>% ggplot(aes(x=GROUP, y=BMI))+ geom_boxplot()+  stat_summary(fun.y=mean, colour="blue", geom="point", shape=2, size=3)+theme_classic(base_size=8)#+ scale_y_continuous(limits = c(0.5, 0.7))
res.aov <- aov(BMI ~ GROUP, data = data_filter)
summary(res.aov)
TukeyHSD(res.aov)
#MEDIE 
app1<- data_filterCTRL$BMI
app2<-data_filterT2DM$BMI
app3 <-data_filterMCV$BMI
a<-mean(app1, na.rm=TRUE )
d<-sd(app1, na.rm=TRUE)
b<-mean(app2, na.rm=TRUE )
e<-sd(app2, na.rm=TRUE)
c<-mean(app3, na.rm=TRUE )
f<-sd(app3, na.rm=TRUE)
table_mean <- matrix(c(a,b,c,d,e,f), ncol=2)
colnames(table_mean)<- c("mean", "sd")
rownames(table_mean) <- c("CTRL","T2DM", "T2DM+MCV")
table_mean
#bxBMI

#box plot gruppi+anova sys
bxsys<-data_filter%>% ggplot(aes(x=GROUP, y=PRESSIONE_SISTOLICA))+ geom_boxplot()+  stat_summary(fun.y=mean, colour="blue", geom="point", shape=2, size=3)+theme_classic(base_size=8)#+ scale_y_continuous(limits = c(0.5, 0.7))
res.aov <- aov(SISTOLIC_PRESSURE ~ GROUP, data = data_filter)
summary(res.aov)
TukeyHSD(res.aov)
#MEDIE 
app1<- data_filterCTRL$PRESSIONE_SISTOLICA
app2<-data_filterT2DM$PRESSIONE_SISTOLICA
app3 <-data_filterMCV$PRESSIONE_SISTOLICA
a<-mean(app1, na.rm=TRUE )
d<-sd(app1, na.rm=TRUE)
b<-mean(app2, na.rm=TRUE )
e<-sd(app2, na.rm=TRUE)
c<-mean(app3, na.rm=TRUE )
f<-sd(app3, na.rm=TRUE)
table_mean <- matrix(c(a,b,c,d,e,f), ncol=2)
colnames(table_mean)<- c("mean", "sd")
rownames(table_mean) <- c("CTRL","T2DM", "T2DM+MCV")
table_mean
#bxsys
#grid.arrange(bxHbA1C, bxHDL,bxTG, bxTGHDL) 
#grid.arrange(bxAGE,bxBMI,bxsys) 

