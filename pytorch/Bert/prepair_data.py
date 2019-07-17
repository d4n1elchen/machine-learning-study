import os
import pandas as pd

DATA_FOLDER = "../../datasets/ainimal_offline_event_data_20190127"

SCORE_CSV = os.path.join(DATA_FOLDER, "score.csv")

PERSONALITIES = ["睿智", "狂野", "活力", "佛系", "浪漫", "善良", "幽默"]

score_df = pd.read_csv(SCORE_CSV)

all_data = pd.DataFrame()

for idx, row in score_df.iterrows():
    id = row["ID"]

    sentence_txt = os.path.join(DATA_FOLDER, "sentences", f"{id:02d}.txt")
    sentence_df = pd.read_csv(sentence_txt, names=["sentence"])

    for pers in PERSONALITIES:
        sentence_df[pers] = row[pers]

    all_data = all_data.append(sentence_df)

all_data.to_csv("data.csv", index=False)
