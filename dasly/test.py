import os
import yaml
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
import time
import pandas as pd
import uuid

db_username = os.getenv('POSTGRESQL_USERNAME')
db_password = os.getenv('POSTGRESQL_PASSWORD')
# print(db_username)
# print(db_password)


# Define the path to the YAML file
yaml_path = 'config.yml'

# Open and read the YAML file
with open(yaml_path, 'r') as file:
    params = yaml.safe_load(file)


# Access parameters from the YAML file
input_dir = params['input_dir']
start_exact_second = params['start_exact_second']
integrate = params['integrate']

database_type = params['database']['database_type']
dbapi = params['database']['dbapi']
endpoint = params['database']['endpoint']
port = params['database']['port']
database = params['database']['database']
database_table = 'test'


def create_connection_string() -> str:
    """Create the connection string for the SQL database."""
    connection_string = (
        f'{database_type}+{dbapi}://{db_username}:{db_password}'
        + f'@{endpoint}:{port}/{database}'
    )
    return connection_string


connection_string = create_connection_string()

for i in range(1, 100):
    print(i)
    engine = create_engine(connection_string, poolclass=NullPool)
    # conn = engine.connect()
    a = str(uuid.uuid4())
    b = str(uuid.uuid4())
    c = str(uuid.uuid4())
    df = pd.DataFrame({
        'a': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
        'b': [4, 5, 6, 1, 2, 3, 1, 2, 3, 1, 2, 3],
        'c': [a, b, c, 1, 2, 3, 1, 2, 3, 1, 2, 3]})
    df.to_sql(database_table, engine, if_exists='append', index=True)
    time.sleep(0.1)
    # conn.close()
    # db.dispose()
