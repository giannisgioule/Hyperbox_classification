*===============================================================================
*                         optimal_hyperbox_model.gms
*
* This script contains the mathematical formulation for fitting hyperboxes to
* a set of multivariate, multilabel classification data. This formulation is
* derived from the following publication:
*
* Xu, G. and Papageorgiou, L.G. (2009). A mixed integer optimisation model for
* data classification. Computers & Industrial Engineering, 56, 1205-1215
*===============================================================================




*=============================Use multiple threads==============================
$onecho>cplex.opt
threads 12
$offecho
*===============================================================================


*===============================================================================
*---------------------------------BEGIN SCRIPT----------------------------------
*===============================================================================

*--------------------------------DEFINE THE SETS--------------------------------
sets
s        The set of samples
m        The set of input variables
i        The set of hyper-boxes
map      Mapping of samples to hyper-boxes;
*-------------------------------------------------------------------------------

*----------------------DEFINE THE CLASSIFICATION PARAMETERS---------------------
parameters
A(s,m)   Value of sample s on attribute m
*-------------------------------------------------------------------------------

*--------------------LOAD THE GDX FILE GENERATED FROM PYTHON--------------------
$GDXIN "input.gdx"
$LOAD  s m i A map
$GDXIN
*-------------------------------------------------------------------------------

alias (i,j);

scalar
U        big number /1.5/;

*--------------------------DEFINE THE MODEL VARIABLES---------------------------
variables
objective        misclassified samples
x(i,m)           Central coordinate of hyper-box i on m;

positive variables

LE(i,m)          Length of hyper-box i on m
;

binary variables
E(s)             1 if sample s is included in the corresponding hyper-box
Y(i,j,m)         0 if box i and j do not overlap each other on attribute m;
*-------------------------------------------------------------------------------

*----------------------------DEFINE MODEL EQUATIONS-----------------------------
equations
eq1      Hyperbox enclosing constraint
eq2      Hyperbox enclosing constraint
eq3      Non-overlapping constraint
eq4      Non-overlapping constraint
obj      Minimisation of misclassified samples;
*-------------------------------------------------------------------------------

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

model data_mining /all/;


option mip=cplex;
option optcr=0.0;
option reslim=200;

solve  data_mining using MIP minimizing objective;

parameter misclassified;

misclassified(s)=(E.l(s)+eps)$(E.l(s)=0);

display objective.l,LE.l,x.l,E.l,Y.l,misclassified;

execute_unload "results.gdx",x.l,LE.l ;
*===============================================================================
*----------------------------------END SCRIPT-----------------------------------
*===============================================================================
