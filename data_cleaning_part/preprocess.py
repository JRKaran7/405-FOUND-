import pandas

def loading_datasets():
    f_n_f = pandas.read_csv("relevant_datasets/food_nutrient.csv", low_memory=False) # Does not infer data types for each column
    n_f = pandas.read_csv("relevant_datasets/nutrient.csv")
    f_f_d = pandas.read_csv("relevant_datasets/food_name_with_description.csv")
        
    f_n_f = f_n_f.drop_duplicates(subset=["fdc_id", "nutrient_id"])
    n_f = n_f.drop_duplicates(subset=["nutrient_nbr"])
    f_f_d = f_f_d.drop_duplicates(subset=["fdc_id"])
        
    f_f_d['fdc_id'] = f_f_d['fdc_id'].astype(int)
    f_f_d = f_f_d[['fdc_id', 'description']]
    f_f_d.rename(columns={'description': 'food'}, inplace=True)
        
    n_cols = ['amount', 'data_points', 'derivation_id', 'min', 'max', 'median', 'min_year_acquired']
    for c in n_cols:
        if c in f_n_f.columns:
            f_n_f[c] = pandas.to_numeric(f_n_f[c], errors='coerce')
        
    if 'amount' in f_n_f.columns:
        f_n_f['amount'] = f_n_f['amount'].fillna(0)

    n_f["nutrient_nbr"] = pandas.to_numeric(n_f["nutrient_nbr"], errors="coerce")
    n_f = n_f.dropna(subset=["nutrient_nbr"])
        
    f_n_f['fdc_id'] = f_n_f['fdc_id'].astype(int)
    f_n_f["nutrient_id"] = f_n_f["nutrient_id"].astype(int)
    n_f["id"] = n_f["id"].astype(int)
        
    m_data = f_n_f.merge(n_f[['id', 'name', 'unit_name']], left_on="nutrient_id", right_on="id", how="left").merge(f_f_d, on="fdc_id", how="left")
        
    m_data = m_data[['fdc_id', 'food', 'nutrient_id', 'name', 'amount', 'unit_name']]
    m_data.rename(columns={'name': 'nutrient_name'}, inplace=True)
        
    m_data = m_data.dropna(subset=["food", "nutrient_name", "amount"])
    m_data = m_data[m_data['amount'] > 0]
        
    return m_data
        