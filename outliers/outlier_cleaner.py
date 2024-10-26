#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []
    
    ### your code goes here
    import pandas as pd
    import numpy as np
    
    errors = np.abs(predictions - net_worths).flatten()
    df = pd.DataFrame({
        'ages':ages.flatten(), 
        'net_worths':net_worths.flatten(), 
        'errors': errors})
    
   # Ordina il dataframe in base all'errore in ordine crescente
    df_sorted = df.sort_values(by='errors')

    # Mantieni solo l'80-90% dei punti con gli errori minori
    cleaned_data = df_sorted.head(int(len(df_sorted) * 0.9))

    # Converte il dataframe in una lista di tuple (age, net_worth, error)
    cleaned_data = list(cleaned_data.itertuples(index=False, name=None))
    
    
    
    return cleaned_data