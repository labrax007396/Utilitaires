import pandas as pd

def ExportPandasSerie_To_UV(data, file_res, desc, ident, mgl, unit, freq, role, MeasureDataType):

    Description = ['Description',desc]
    TagName = ['Tagname',ident] 

    Units = ['Unit',unit]
    freq_list = ['TagInfoFrequency',freq]
    ParentMangling = ['ParentScopeMangling',mgl]
    Role = ['TagInfoRole',role]
    MeasureDataType = ['MeasureDataType', MeasureDataType]     



    data = data.to_frame()
    data = data.reset_index(level=0)
    data.columns = pd.MultiIndex.from_tuples(zip(Description,TagName,ParentMangling,Units,freq_list,Role,MeasureDataType))

    data.to_csv(file_res, date_format='%Y/%m/%d %H:%M:%S',index=False,sep=',')

def ExportPandasDf_To_UV(data,rep,fichier,parentMangling,freq,role,dtype,**kwargs):

    import os.path

    if 'cols_to_export' in kwargs.keys():
        cols_to_export = kwargs.get('cols_to_export', None)
        col_data = list(data.columns)
        for ctexp in cols_to_export:
                if ctexp not in col_data:
                    print(ctexp + ' absente')
                    return None
    else:
        cols_to_export = list(data.columns)

    if 'Unit' in kwargs.keys():
        unit = kwargs.get('Unit', None)
        if type(unit) == str:
                Unit = [unit]*len(cols_to_export)
        else:
                if len(unit) != len(cols_to_export):
                    print('Le nombre unité '+str(len(unit))+ ' ne match pas avec le nombre de variables: '+str(len(cols_to_export)))
                    return None
                else:
                    Unit = unit

    if type(parentMangling) == str:
        ParentMangling = [parentMangling]*len(cols_to_export)
    else:
        if len(parentMangling) != len(cols_to_export):
                print('Le nombre de parentmgl: '+str(len(parentMangling))+ ' ne match pas avec le nombre de variables: '+str(len(cols_to_export)))
                return None
        else:
                ParentMangling = parentMangling     


    if type(freq) == str:
        Freq = [freq]*len(cols_to_export)
    else:
        if len(freq) != len(cols_to_export):
                print('Le nombre de freq: '+str(len(freq))+ ' ne match pas avec le nombre de variables: '+str(len(cols_to_export)))
                return None
        else:
                Freq = freq    
    
    if type(role) == str:
        Role = [role]*len(cols_to_export)
    else:
        if len(role) != len(cols_to_export):
                print('Le nombre de role: '+str(len(freq))+ ' ne match pas avec le nombre de variables: '+str(len(cols_to_export)))
                return None
        else:
                Role = role

    if type(dtype) == str:
        Dtype = [dtype]*len(cols_to_export)
    else:
        if len(dtype) != len(cols_to_export):
                print('Le nombre de type de données: '+str(len(dtype))+ ' ne match pas avec le nombre de variables: '+str(len(cols_to_export)))
                return None
        else:
                Dtype = dtype

    if 'tagname' in kwargs.keys():  
        tagname = kwargs.get('tagname', None)    
        if len(tagname) != len(cols_to_export):
                print('Le nombre de tagname '+str(len(tagname))+ ' ne match pas avec le nombre de variables: '+str(len(cols_to_export)))
                return None
        else:
                Tagname = tagname

    else:
        Tagname = cols_to_export

    Description    = ['Description']+cols_to_export
    TagName        = ['Tagname']+Tagname
    Units          = ['Unit']+Unit
    freq_list      = ['TagInfoFrequency']+Freq
    ParentMangling = ['ParentScopeMangling']+ParentMangling
    Role           = ['TagInfoRole']+Role
    MeasureDataType= ['MeasureDataType']+Dtype


    data_to_export = data[cols_to_export]

    data_to_export.insert(loc=0,column='Date',value=data_to_export.index)


    data_to_export.columns = pd.MultiIndex.from_tuples(zip(Description,TagName,ParentMangling,Units,freq_list,Role,MeasureDataType))

    if not os.path.exists(rep) :
        print("Chemin " , rep, " n'existe pas")
        return
    else:
        data_to_export.to_csv(rep+fichier, date_format='%Y/%m/%d %H:%M:%S',index=False,sep=',')
     
