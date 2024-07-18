import os

db_username = os.getenv('POSTGRESQL_USERNAME')
db_password = os.getenv('POSTGRESQL_PASSWORD')


print(f'Username: {db_username}')
print(f'Password: {db_password}')