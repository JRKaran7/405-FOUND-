import pandas

def load():
    food_nutrient_df = pandas.read_csv("datasets/food_nutrient.csv", low_memory=False)
    nutrient_df = pandas.read_csv("datasets/nutrient.csv")
    foundation_food_df = pandas.read_csv("datasets/foundation_food.csv")

    col1 = ["fdc_id", "nutrient_id"]
    col2 = ["nutrient_nbr"]
    col3 = ["fdc_id"]
    food_nutrient_df = food_nutrient_df.drop_duplicates(subset=col1)
    nutrient_df = nutrient_df.drop_duplicates(subset=col2)
    foundation_food_df = foundation_food_df.drop_duplicates(subset=col3)

    columns = ['amount', 'data_points', 'derivation_id', 'min', 'max', 'median', 'min_year_acquired']
    for col in columns:
        if col in food_nutrient_df.columns:
            food_nutrient_df[col] = pandas.to_numeric(food_nutrient_df[col], errors='coerce').fillna(food_nutrient_df[col].mean())

    nutrient_df["nutrient_nbr"] = pandas.to_numeric(nutrient_df["nutrient_nbr"], errors="coerce")
    nutrient_df = nutrient_df.dropna(subset=["nutrient_nbr"])

    food_nutrient_df['fdc_id'] = food_nutrient_df['fdc_id'].astype(int)
    foundation_food_df['fdc_id'] = foundation_food_df['fdc_id'].astype(int)

    food_nutrient_df["nutrient_id"] = food_nutrient_df["nutrient_id"].astype(int)
    nutrient_df["nutrient_nbr"] = nutrient_df["nutrient_nbr"].astype(int)
    
    foundation_food_df["food"] = foundation_food_df["NDB_number"].astype(str) + " " + foundation_food_df["footnote"].fillna(" ")

    merged_dataset = food_nutrient_df.merge(nutrient_df, left_on="nutrient_id", right_on="id", how="left").merge(foundation_food_df[["fdc_id", "food"]], on="fdc_id", how="left")

    merged_dataset = merged_dataset[['fdc_id', 'food', 'nutrient_id', 'name', 'amount', 'unit_name']]
    merged_dataset.rename(columns={'name': 'nutrient_name'}, inplace=True)

    merged_dataset = merged_dataset.dropna(subset=["food", "nutrient_name", "amount"])

    return merged_dataset
