def generate_gdx(dataset):
    
    #================================================================
    # Import the required libraries.    
    #================================================================    
    import gdxpds as gd
    import pandas as pd
    import numpy as np    
    #================================================================    
            
    number_predictors=dataset.shape[1]-1 # The number of predictors in the set
    attribute_names=dataset.columns[0:number_predictors] # The names of the attributes        

    # Create the set of samples
    s_val=list(range(1, dataset.shape[0]+1))
    s_val=list(map(str,s_val))
    s_val=["s"+x for x in s_val]

    # Create the list parameter for the input values
    s_val=s_val*number_predictors
    s_val=pd.DataFrame(s_val)

    values=dataset.iloc[:,0:number_predictors]
    values=values.unstack()
    values=pd.DataFrame(list(values))

    all_attr=pd.DataFrame(np.repeat(attribute_names[0],dataset.shape[0]))
    for i in range(1,number_predictors):
        single_attr=pd.DataFrame(np.repeat(attribute_names[i],dataset.shape[0]))
        all_attr=pd.concat([all_attr,single_attr],axis=0)

    all_attr.index=s_val.index
    final_values=pd.concat([s_val,all_attr],axis=1)
    final_values=pd.concat([final_values,values],axis=1)
    del(values)

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
    
    input_data[0].dataframe=s_val[0:dataset.shape[0]] # Define the samples set
    input_data[1].dataframe=attribute_names # Define the input set
        
    input_data[2].dataframe=final_values
    
    dataset.index=s_val[0:dataset.shape[0]].index
    input_data[3].dataframe=pd.concat([s_val[0:dataset.shape[0]],dataset.iloc[:,number_predictors]],axis=1)
    
    # Identify the different number of classes in the label
    input_data[4].dataframe=pd.DataFrame(output) 
    
    input_data.write(gdx_file)
    
    return(".gdx file has been generated")

