from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd
import numpy as np
import os        
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.multiclass import unique_labels
from pyomo.environ import *
from pyomo.opt import SolverFactory
from scipy.spatial import distance
from operator import itemgetter

class HyperboxClassifier():
    '''
    A Hyperbox classifier

    This classifier creates hypeboxes, each one representing a single class in the data. The 
    algorithm optimizes the coordinates and dimensions of the hyperboxes in order to minimize
    the number of misclassified samples

    Parameters
    ----------
    NONE

    Attributes
    ----------

    model_: dict

    A dictionary that contains the following information for each box:

    - LE: The length of the hyperbox for each dimension 
    - x : The central coordinates of each hyperbox for each dimension
    - vertces: The coordinates of every vertex of the hyperbox for each dimension
    
    '''

    def __init__(self):
        pass

    #===============================Fit method================================    
    def fit(self,X,y):

        class Boxes():

            def __init__(self,LE,x,vertices=0):
                self.LE=LE
                self.x=x
                self.vertices=vertices

        #---------------------------------------------------------------------
        def define_vertices(LE,x):
            # This function requires the LE (length) and x (central coordinates)
            # of each hyperbox and calculates the coordinates of all the vertices
            number_of_boxes=np.unique(LE['i'])
            vertices={}
            for box in number_of_boxes:
                lower_limits=list(x[x['i']==box]['Level']-(LE[LE['i']==box]\
                    ['Level']/2))
                upper_limits=list(x[x['i']==box]['Level']+(LE[LE['i']==box]\
                    ['Level']/2))
                vertices.update({box:([lower_limits],[upper_limits])})
                
            return vertices
        #---------------------------------------------------------------------            


        #---------------------------------------------------------------------
        # This function generates the data in an appropriate format to 
        # be included in pyomo. This function converts the class names
        # into numbers to be processed by pyomo. The original names are
        # substituted back again in a later stage
        def generate(dataset):


            # The number of predictors in the set
            number_predictors=dataset.shape[1]-1 
            
            # The names of the attributes      
            attribute_names=dataset.columns[0:number_predictors]   

            #--------------------Create the set of samples--------------------
            s_val=list(map(str,dataset.index))
            s_val=["s"+x for x in s_val]
            s=s_val
            #-----------------------------------------------------------------

            #-----------------Define variable and class names-----------------
            m=attribute_names
            original_boxes=list(np.unique(dataset.iloc[:,number_predictors]))
            boxes=list(range(len(original_boxes)))
            #-----------------------------------------------------------------
            
            #-------------Create the appropriate format for pyomo-------------
            s_val=s_val*number_predictors
            s_val=pd.DataFrame(s_val)

            values=dataset.iloc[:,0:number_predictors]
            values=values.unstack()
            values=pd.DataFrame(list(values))

            all_attr=pd.DataFrame(np.repeat(attribute_names[0],\
                dataset.shape[0]))
            for i in range(1,number_predictors):
                    single_attr=pd.DataFrame(np.repeat(attribute_names[i],\
                        dataset.shape[0]))
                    all_attr=pd.concat([all_attr,single_attr],axis=0)                

            all_attr.index=s_val.index
            final_values=pd.concat([s_val,all_attr],axis=1)
            final_values=pd.concat([final_values,values],axis=1)
            del(values)
            
            A={}    
            for i in range(final_values.shape[0]):
                    A.update({tuple(final_values.iloc[i,0:2])\
                        :final_values.iloc[i,2]})
            #-----------------------------------------------------------------

            #---------------------Create the mapping set----------------------
            # This set maps which samples belong to which hyperboxes
            
            dataset.index=s_val[0:dataset.shape[0]].index
            map_samples=pd.concat([s_val[0:dataset.shape[0]],dataset.iloc\
                [:,number_predictors].replace(original_boxes,boxes)],axis=1)
                        
            mapping={}
            
            for i in range(map_samples.shape[0]):
                    mapping.update({tuple(map_samples.iloc[i,:]):1})
            #-----------------------------------------------------------------

            data={None:{
            'i':{None:boxes},
            'm':{None:m},
            's':{None:s},
            'A':A,
            'map':mapping
            }}

            return data,original_boxes,boxes            
        #---------------------------------------------------------------------
        

        #----------------------Pyomo optimisation model-----------------------
        def pyomo_model(data,boxes,original_boxes):
            
            model=AbstractModel()
            
            
            #----------------------Define the model sets----------------------            
            model.s=Set(dimen=1,doc='Samples')
            model.m=Set(dimen=1,doc='Features')
            model.i=Set(dimen=1,doc='Boxes')
            model.j=SetOf(model.i)
            #-----------------------------------------------------------------

            
            #------------Define the model parameters and variables------------            
            # Parameters
            model.A=Param(model.s,model.m,doc='Value of sample s on attribute m')
            model.U=Param(initialize=1.5)
            model.map=Param(model.s,model.i,doc='Mapping',default=0)

            # Positive Variables
            model.x=Var(model.i,model.m,domain=NonNegativeReals)
            model.LE=Var(model.i,model.m,domain=NonNegativeReals)

            # Binary Variables
            model.E=Var(model.s,domain=Binary)
            model.Y=Var(model.i,model.j,model.m,domain=Binary)
            #-----------------------------------------------------------------

            #-------------------------Model equations-------------------------
            
            # Objective Function - Minimize misclassifications
            def objective(model):
                    objective_value=sum(1-model.E[s] for s in model.s)
                    return objective_value
            model.OBJ=Objective(rule=objective,sense=minimize)

            # Enclosing constraint 1 - Samples should fall within box limits
            def enclosing_1(model,i,s,m):
                    if(model.map[s,i]==1):
                            return model.A[s,m]>=model.x[i,m]-\
                                (model.LE[i,m]/2)-model.U*(1-model.E[s])
                    else:
                            return Constraint.Skip
            model.eq1=Constraint(model.i,model.s,model.m,rule=enclosing_1)

            # Enclosing constraint 2 - Samples should fall within box limits
            def enclosing_2(model,i,s,m):
                    if(model.map[s,i]==1):
                            return model.A[s,m]<=model.x[i,m]+\
                                (model.LE[i,m]/2)+model.U*(1-model.E[s])
                    else:
                            return Constraint.Skip
            model.eq2=Constraint(model.i,model.s,model.m,rule=enclosing_2)

            # Non-overlapping constraint 1 - Boxes should not overlap
            def non_overlap_1(model,m,i,j):
                    if(i!=j):
                            return model.x[i,m]-model.x[j,m]+\
                                model.U*model.Y[i,j,m]>=\
                                    (model.LE[i,m]+model.LE[j,m])/2+0.001
                    else:
                            return Constraint.Skip
            model.eq3=Constraint(model.m,model.i,model.j,rule=non_overlap_1)

            # Non-overlapping constraint 2 - Boxes should not overlap
            def non_overlap_2(model,i,j):
                    if(i<len(model.i) and j>=i+1):
                            return sum(model.Y[i,j,m]+model.Y[j,i,m]\
                                for m in model.m)<=2*len(model.m)-1
                    else:
                            return Constraint.Skip
            model.eq4=Constraint(model.i,model.j,rule=non_overlap_2)                        
            #-----------------------------------------------------------------

            #-------------------------Solver options--------------------------
            # Solver - CPLEX (the user has to manually install the solver)
            # Optimality gap - 0%
            # Timelimit - 200s
            opt=SolverFactory('cplex')
            opt.options['mipgap']=0
            opt.options['timelimit'] = 200            
            instance=model.create_instance(data)
            opt.solve(instance,tee=True)            
            #-----------------------------------------------------------------

            #---------------------Results post-processing---------------------
            x=pd.DataFrame()
            LE=pd.DataFrame()
            for index in instance.x:
                x_decoy=pd.concat([pd.DataFrame([index[0]]),\
                    pd.DataFrame([index[1]]),\
                        pd.DataFrame([instance.x[index].value])],axis=1)
                x=pd.concat([x,x_decoy],axis=0)
            x.columns=["i","m","Level"]
            for index in instance.LE:
                LE_decoy=pd.concat([pd.DataFrame([index[0]]),\
                    pd.DataFrame([index[1]]),\
                        pd.DataFrame([instance.LE[index].value])],axis=1)
                LE=pd.concat([LE,LE_decoy],axis=0)
            LE.columns=["i","m","Level"]   
            
            x["i"]=x["i"].replace(boxes,original_boxes)
            LE["i"]=LE["i"].replace(boxes,original_boxes)
            #-----------------------------------------------------------------

            return x,LE    
        #---------------------------------------------------------------------  
        
        # By default, the algorithm scales the input variables in the range
        # [0,1]. This is done in order to aid the optimisation and the bigM
        # constraints.

        # So, at the end of the optimisation, the regression coefficients
        # and intercepts of each region are scaled back to reflect the 
        # original data.                
        
        #---------------------------------------------------------------------
        # This method unscales the values of the central coordinates (x) 
        # of the hyperboxes
        def unscale_values(vector):    
            variables=np.unique(vector['m'])
            for i,j in enumerate(variables):
                b={}
                to_replace=vector['Level'][vector['m']==j]
                value=(vector['Level'][vector['m']==j]*(scaler.data_max_[i]-\
                    scaler.data_min_[i])+scaler.data_min_[i])
                [b.update({i:j}) for i,j in zip(to_replace,value)]
                vector['Level']=vector['Level'].replace(to_replace=b,value=None)
                
            return vector
        #---------------------------------------------------------------------                      
        
        #---------------------Main part of the fit method---------------------        
        v_names=False
        if(isinstance(X,pd.DataFrame)):
            variable_names=X.columns
            v_names=True
                
        o_name=False
        if(isinstance(y,pd.DataFrame)):
            output_name=y.columns
            o_name=True
        
        X,y=check_X_y(X,y)
        scaler=MinMaxScaler(feature_range=(0,1))
        scaler.fit(X)
        X=scaler.transform(X)                        
        X=pd.DataFrame(X)        
        y=pd.DataFrame(y)                
        if(v_names is True):
            X.columns=variable_names        
        else:
            # If there are no names, create a vector for the names.
            # x0,x1,x2,...
            X.columns=[("x"+str(i)) for i in range(len(X.columns))]
        if(o_name is True):
            y.columns=output_name
        if(v_names is True):        
            self.names_=variable_names
        else:
            self.names_=False          
        
        self.classes_=unique_labels(y)                
        X=pd.concat([X,y],axis=1)                

        input_data,original_boxes,boxes=generate(X)
        x,LE=pyomo_model(input_data,boxes,original_boxes)                        
        x=unscale_values(x)

        hyperboxes=Boxes(LE,x)
        box_vertices=define_vertices(hyperboxes.LE,hyperboxes.x)
        hyperboxes.vertices=box_vertices

        self.is_fitted_=True
        self.model_=hyperboxes

        return self
        #---------------------------------------------------------------------
    # End of fit method
    #=========================================================================

    #=============================Predict method==============================
    def predict(self,X):        

        # For each sample, the prediction checks if the values of the sample
        # are in the range of one of the hyperboxes.
                
        X=check_array(X)
        check_is_fitted(self,'is_fitted_')  

        names=getattr(self,"names_")

        X=pd.DataFrame(X)
        if(np.all(names is not False)):
            X.columns=names                
            names=list(names)
        else:        
            X.columns=[("x"+str(i)) for i in range(len(X.columns))]
            names=[("x"+str(i)) for i in range(len(X.columns))]  

        hyperboxes=getattr(self,"model_")
               
        def identify_class(X_test,vertices,box_centres):

            number_predictors=X_test.shape[1] 
            testing_samples=X_test.iloc[:,0:number_predictors]
            testing_samples=testing_samples.values.tolist()

            testing_class=[]

            for testing in range(len(testing_samples)):

                number_of_boxes=vertices.keys()    
                for i in number_of_boxes: 
                    a=[testing_samples[testing]]
                    b=vertices[i][0]
                    c=vertices[i][1]     
                    # Check if the values are in the ranges of the hyperboxes  
                    if((sum([ x>=y for (x,y) in zip(a[0], b[0])])==\
                        number_predictors) and (sum([ x<=y for (x,y) \
                            in zip(a[0], c[0])])==number_predictors)):  
                        testing_class.append(i)
                        break    
                    else:
                        # Unassigned samples will be assigned to the closest hyperbox.
                        # The closest hyperbox is the one with the min euclidean distance
                        # between its' centre and the sample                    
                        testing_class.append(unassigned_samples\
                            (testing_samples[testing],\
                                box_centres,number_predictors))
                        break

            return testing_class

        def unassigned_samples(testing_sample,box_centres,number_predictors):        

            classes=np.unique(box_centres["i"])
            classes.sort()

            sorted_centres=box_centres.sort_values(['i'])['Level']        
            sorted_centres=sorted_centres.values.tolist()

            sorted_centres=[sorted_centres[x:x+number_predictors] \
                for x in range(0,len(sorted_centres),number_predictors)]
            min_distance=[distance.euclidean(testing_sample,sorted_centres[x])\
                 for x in range(0,len(sorted_centres))]

            index_of_class=min(enumerate(min_distance), key=itemgetter(1))[0] 
            box=classes[index_of_class]

            return box

        predictions=identify_class(X,hyperboxes.vertices,hyperboxes.x)                

        return predictions
    # End of printing equations
    #=========================================================================                