*============= Use multiple threads==============
$onecho>cplex.opt
threads 12
$offecho
*================================================


*===============================================================================
*-----------------------------Load the input data-------------------------------
*===============================================================================
*-----------------------2 Features, 399 samples, 4 classes----------------------
*$set dataset compound

*-----------------------2 Features, 300 samples, 3 classes----------------------
*$set dataset pathbased

*-----------------------2 Features, 788 samples, 7 classes----------------------
*$set dataset aggregation

*----------------------2 Features, 1000 samples, 2 classes----------------------
*$set dataset synthetic_1

*----------------------2 Features, 1000 samples, 2 classes----------------------
*$set dataset synthetic_2

$set dataset input

*===============================================================================

*===============================================================================
*----------------------------------BEGIN MODEL----------------------------------
*===============================================================================


sets
s        Samples
m        Attribute
i        Hyper-boxes
map      Mapping of samples;


parameters
A(s,m)   Value of sample s on attribute m

$GDXIN "%dataset%.gdx"
$LOAD  s m i A map
$GDXIN

alias (i,j);

display s,m,i,A,map;

scalar
U        big number /1.5/;

variables

objective misclassified samples
x(i,m)   Central coordinate of hyper-box i on m;

positive variables

LE(i,m)  Length of hyper-box i on m
;

binary variables
E(s)     1 if sample s is included in the corresponding hyper-box
Y(i,j,m) 0 if box i and j do not overlap each other on attribute m;

equations
eq1      Hyperbox enclosing constraint
eq2      Hyperbox enclosing constraint
eq3      Non-overlapping constraint
eq4      Non-overlapping constraint
obj      Minimisation of misclassified samples;

eq1(i,s,m)$(map(s,i))..
         A(s,m)=g=x(i,m)-(LE(i,m)/2)-U*(1-E(s));

eq2(i,s,m)$(map(s,i))..
         A(s,m)=l=x(i,m)+(LE(i,m)/2)+U*(1-E(s));

eq3(m,i,j)$(ord(i) ne ord(j))..
         x(i,m)-x(j,m)+U*Y(i,j,m)=g=((LE(i,m)+LE(j,m))/2)+0.001;

eq4(i,j)$((ord(i) < card(i))and(ord(j) ge ord(i)+1))..
         sum(m,(Y(i,j,m)+Y(j,i,m)))=l=2*card(m)-1;

obj..
         objective=e=sum(s,(1-E(s)));

model data_mining /eq1,eq2,eq3,eq4,obj/;

*===============================================================================
* 200s CPU limit, 0% optimal gap
*===============================================================================
option mip=cplex;
option optcr=0.0;
option Limrow =100;
option reslim=200;
*===============================================================================
E.fx('s7')=1;
solve  data_mining using MIP minimizing objective;

parameter misclassified;

misclassified(s)=(E.l(s)+eps)$(E.l(s)=0);

display objective.l,LE.l,x.l,E.l,Y.l,misclassified;

execute_unload "results.gdx",x.l,LE.l ;
*===============================================================================
*-----------------------------------END MODEL-----------------------------------
*===============================================================================
