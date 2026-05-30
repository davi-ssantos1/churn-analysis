import pandas as pd
from pydantic import ValidationError

from churn_analysis.config import RAW_DATA_PATH
from churn_analysis.ETL.schemas import CustomRecord


def extraction() -> pd.DataFrame:
    "Extract the csv file and clean it base on the schemas, transforming it into a DataFrame"
    csv_reader = pd.read_csv(RAW_DATA_PATH, chunksize=1000)
    clean_chunks = []
    for df_chunk_index, df_chunk in enumerate(csv_reader):
        records = df_chunk.to_dict(orient="records")

        for row_index, row in enumerate(records):
            try:
                custom_clean_record = CustomRecord(**row)
                dict_clean_record = custom_clean_record.model_dump()
                clean_chunks.append(dict_clean_record)
            except ValidationError:
                raise Exception(
                    f"Corrupted data from {df_chunk_index * 1000 + row_index + 1} row.\nRow: {row}"
                ) from None

    return pd.DataFrame(clean_chunks)


if __name__ == "__main__":
    print(extraction())
