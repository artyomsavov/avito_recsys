import duckdb

train_query = "SELECT count(*) FROM 'data/raw/train_data/*.parquet'"
train_count = duckdb.sql(train_query).fetchone()[0]
print(f"Событий в train: {train_count:,}")

eval_query = "SELECT count(*) FROM read_parquet('data/raw/eval_user_events.pq')"
eval_count = duckdb.sql(eval_query).fetchone()[0]
print(f"Событий в eval:  {eval_count:,}")

print(f"Всего событий:   {train_count + eval_count:,}")
