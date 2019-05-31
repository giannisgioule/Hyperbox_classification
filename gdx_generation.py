#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def generate_gdx(dataset):
    
    #================================================================
    # Import the required libraries.    
    #================================================================    
    import gdxpds as gd
    import pandas as pd
    import numpy as np
    import os
    #================================================================    
            
    number_predictors=dataset.shape[1]-1 # The number of predictors in the set
    attribute_names=dataset.columns[0:number_predictors] # The names of the attributes        
    
    s_val=pd.DataFrame(list(range(1, dataset.shape[0]+1))) # The set of samples 
    out=dataset.iloc[:,number_predictors] # The output variable
    out.index=s_val.index # Change the row names of the set

    values=dataset.iloc[:,0] # The input of the dataset

    for x_index in range(0,s_val.shape[0]):
        s_val.iloc[x_index,0]="s"+str(s_val.iloc[x_index,0])
    del(x_index)

    # Create the input of regression with the appropriate GAMS format
    
    #================================================================
    # Creating the appropriate format for gdx. A two dimensional 
    # dataframe, has to be converted from a convetional table format.
    # 
    # ss_val:         duplicate of s_val
    # single_attr:    one dimensional dataframe containing a single 
    #                 variable
    # all_attr:       one dimensional dataframe containing all the
    #                 variables.     
    #================================================================
    
    ss_val=pd.DataFrame(s_val) # the two sets for the input
    
    all_attr=pd.DataFrame(np.repeat(attribute_names[0],dataset.shape[0]))
    all_attr.index=s_val.index # initial values. Choose only the first variable
    
    # Loop for the rest of the variables
    for i in range(1,number_predictors):
        values=pd.concat([values,dataset.iloc[:,i]],axis=0)
        ss_val=pd.concat([ss_val,s_val],axis=0)
        single_attr=pd.DataFrame(np.repeat(attribute_names[i],dataset.shape[0]))
        single_attr.index=s_val.index
        all_attr=pd.concat([all_attr,single_attr],axis=0)
        
    values.index=list(range(0,values.shape[0]))
    ss_val.index=list(range(0,values.shape[0]))

    all_attr.index=ss_val.index

    ss_val=pd.concat([ss_val,all_attr],axis=1)
    values=pd.DataFrame(pd.concat([ss_val,values],axis=1)) 
    del(all_attr,ss_val)
    
    # Capture the different classes of the output
    output=list(np.unique(dataset.iloc[:,number_predictors]))
    
    #================================================================
    # Define the .gdx file and assign values to all the elements.
    #================================================================
    gdx_file="input.gdx" # always remember the path
    
    input_data=gd.gdx.GdxFile() # The variable that will contain all of the gdx sets/variables

    # First define the set of the samples
    input_data.append(gd.gdx.GdxSymbol("s",gd.gdx.GamsDataType.Set,dims=1,description="set of samples in the set"))

    # Then define the set of the predictor variables
    input_data.append(gd.gdx.GdxSymbol("m",gd.gdx.GamsDataType.Set,dims=1,description="The input variables of the dataset"))

    # Then define the input of the dataset
    input_data.append(gd.gdx.GdxSymbol("A",gd.gdx.GamsDataType.Parameter,dims=2,description="The values of the samples"))

    input_data.append(gd.gdx.GdxSymbol("map",gd.gdx.GamsDataType.Set,dims=2,description="Mapping of samples"))
    
    input_data.append(gd.gdx.GdxSymbol("i",gd.gdx.GamsDataType.Set,dims=1,description="Hyper-boxes"))
    
    input_data[0].dataframe=s_val # Define the samples set
    input_data[1].dataframe=attribute_names # Define the input set
        
    input_data[2].dataframe=values
    
    input_data[3].dataframe=pd.concat([s_val,dataset.iloc[:,number_predictors]],axis=1)
    
    # Identify the different number of classes in the label
    input_data[4].dataframe=pd.DataFrame(output) 
    
    input_data.write(gdx_file)
    
    return(".gdx file has been generated")

