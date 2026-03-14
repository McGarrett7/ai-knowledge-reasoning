#bổ sung noisy
import pandas as pd
import random
import rasterio
from sklearn.model_selection import train_test_split

#Vietnam_PVOUT.tif download https://globalsolaratlas.info/download/vietnam
#Chuyển file tif thành csv
def create_dataset():
    dataset = rasterio.open("../PVOUT.tif")
    band = dataset.read(1)
    rows, cols = band.shape
    data = []

    for i in range(rows):
        for j in range(cols):
            value = band[i, j]
            if value > 0:
                data.append((i, j, float(value)))

    df = pd.DataFrame(data, columns=["x","y","pv_potential"])
    df.to_csv("../dataset/solar_dataset.csv", index=False)
    print("CSV created:", len(df), "rows")

def raster_size():
    dataset = rasterio.open("../PVOUT.tif")
    band = dataset.read(1)
    rows, cols = band.shape
    print("Raster size:", rows, cols)
    data = []
    samples = 20000   # số điểm lấy mẫu

    for _ in range(samples):
        i = random.randint(0, rows-1)
        j = random.randint(0, cols-1)
        value = band[i, j]
        if value > 0:
            data.append((i, j, float(value)))

    df = pd.DataFrame(data, columns=["x","y","pv_potential"])
    print("Sampled rows:", len(df))
    df.to_csv("../dataset/solar_dataset.csv", index=False)
    print("CSV created")

    # result: Raster size: 14173 9448
    # Sampled rows: 19757
    # CSV created

def knowledge_graph():
    data = pd.read_csv("../dataset/solar_dataset.csv")
    triples = []

    for _,row in data.iterrows():
        location = f"location_{int(row['x'])}_{int(row['y'])}"
        triples.append((location,"hasSolarPotential",row["pv_potential"]))
        triples.append((location,"produces","solar_energy"))

    kg = pd.DataFrame(triples, columns=["head","relation","tail"])
    kg = kg.sample(5000)

    kg.to_csv("../dataset/renewable_kg.csv", index=False)
    print("Knowledge Graph dataset created")
    # result: Knowledge Graph dataset created

def dataset_split():
    kg = pd.read_csv("../dataset/renewable_kg.csv")
    train, test = train_test_split(kg, test_size=0.2, random_state=42)
    train, val = train_test_split(train, test_size=0.2, random_state=42)
    train.to_csv("../dataset/train.csv", index=False)
    val.to_csv("../dataset/validation.csv", index=False)
    test.to_csv("../dataset/test.csv", index=False)
    print(len(train), len(val), len(test))
    # result: 3200 800 1000

def add_noise():
    kg = pd.read_csv("../dataset/renewable_kg.csv")
    noise_rows = []
    for i in range(300):
        row = kg.iloc[random.randint(0,len(kg)-1)]
        noise_rows.append({
            "head":row["head"],
            "relation":row["relation"],
            "tail":random.randint(100, 300)
        })

    noise = pd.DataFrame(noise_rows)
    kg_noisy = pd.concat([kg,noise])
    kg_noisy.to_csv("../dataset/renewable_kg_noisy.csv",index=False)
    print("Noisy dataset created:",len(kg_noisy))
