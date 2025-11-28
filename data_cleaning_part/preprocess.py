import pandas

def load():
    food_nutrient_dataset = pandas.read_csv("datasets/food_nutrient.csv")
    nutrient_dataset = pandas.read_csv("datasets/nutrient.csv")
    foundation_food_dataset = pandas.read_csv("datasets/foundation_food.csv")

    col1 = ["fdc_id", "nutrient_id"]
    col2 = ["nutrient_nbr"]
    col3 = ["fdc_id"]
    food_nutrient_dataset = food_nutrient_dataset.drop_duplicates(subset=col1)
    nutrient_dataset = nutrient_dataset.drop_duplicates(subset=col2)
    foundation_food_dataset = foundation_food_dataset.drop_duplicates(subset=col3)

    columns = ["amount","data_points","derivation_id","min","max","median","footnote","min_year_acquired"]
    for i in columns:
        food_nutrient_dataset = pandas.to_numeric(food_nutrient_dataset[i]).fillna(food_nutrient_dataset[i].mean())

    foundation_food_dataset["f_name"] = foundation_food_dataset["NDB_number"] + " " + foundation_food_dataset["footnote"].fillna(" ")

    merged_dataset = food_nutrient_dataset.merge(nutrient_dataset, left_on="nutrient_id", right_on="nutrient_nbr", how="left").merge(foundation_food_dataset["fdc_id", "f_name"], on="fdc_id", how="left")

    merged_dataset = merged_dataset.dropna(subset=["f_name", "name", "amount"])

    return merged_dataset