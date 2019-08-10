# https://towardsdatascience.com/deploying-a-machine-learning-model-as-a-rest-api-4a03b865c166

import requests

url = 'http://localhost:5000/'
params = {'query': 'that movie was boring'}
response = requests.get(url, params)
print(response.json())