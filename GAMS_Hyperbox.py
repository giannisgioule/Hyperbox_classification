class GamsHyperboxClassifier():

    def __init__(self):
        pass
    
    def fit(self,X):        
        import gdxpds as gd
        import pandas as pd
        import numpy as np
        import os        

        class Boxes():

            def __init__(self,LE,x,vertices=0):
                self.LE=LE
                self.x=x
                self.vertices=vertices

        # This function calculates the coordinates of the vertices
        # of the hyperboxes. This is usefull for predictions
        def define_vertices(LE,x):
            number_of_boxes=np.unique(LE['i'])
            vertices={}
            for i in number_of_boxes:
                lower_limits=list(x[x['i']==i]['Level']-(LE[LE['i']==i]['Level']/2))
                upper_limits=list(x[x['i']==i]['Level']+(LE[LE['i']==i]['Level']/2))
                vertices.update({i:([lower_limits],[upper_limits])})
                
            return vertices
        def generate_gdx(dataset):                   
                    
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

        # Generate the gdx input file, call GAMS and save
        # the results of the optimisation.
        generate_gdx(X)
        os.system("gams data_mining.gms o=nul")
        results=gd.to_dataframes('results.gdx')

        LE=results['LE']
        x=results['x']
        hyperboxes=Boxes(LE,x)
        box_vertices=define_vertices(hyperboxes.LE,hyperboxes.x)
        hyperboxes.vertices=box_vertices

        # variable hyperboxes has the class attributes of
        # LE: length of each hyperbox for each dimension
        # x: Central coordinate of hyperbox for each dimension
        # vertices: Hyperbox vertices of for each dimension

        return hyperboxes


    def predict(self,hyperboxes,X_test):

        import pandas as pd
        import numpy as np
        from scipy.spatial import distance
        from operator import itemgetter

        # For each sample, the prediction checks if the values of the sample
        # are in the range of one of the hyperboxes.
               
        def identify_class(X_test,vertices,box_centres):

            number_predictors=X_test.shape[1]-1 
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
                    if((sum([ x>=y for (x,y) in zip(a[0], b[0])])==number_predictors) and (sum([ x<=y for (x,y) in zip(a[0], c[0])])==number_predictors)):  
                        testing_class.append(i)
                        break    
                    else:
                        # Unassigned samples will be assigned to the closest hyperbox.
                        # The closest hyperbox is the one with the min euclidean distance
                        # between its' centre and the sample                    
                        testing_class.append(unassigned_samples(testing_samples[testing],box_centres,number_predictors))
                        break

            return testing_class

        def unassigned_samples(testing_sample,box_centres,number_predictors):        

            classes=np.unique(box_centres["i"])
            classes.sort()

            sorted_centres=box_centres.sort_values(['i'])['Level']        
            sorted_centres=sorted_centres.values.tolist()

            sorted_centres=[sorted_centres[x:x+number_predictors] for x in range(0,len(sorted_centres),number_predictors)]
            min_distance=[distance.euclidean(testing_sample,sorted_centres[x]) for x in range(0,len(sorted_centres))]

            index_of_class=min(enumerate(min_distance), key=itemgetter(1))[0] 
            box=classes[index_of_class]

            return box

        predictions=identify_class(X_test,hyperboxes.vertices,hyperboxes.x)

        return predictions