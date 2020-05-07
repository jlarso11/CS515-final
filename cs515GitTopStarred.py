import requests 
from requests.auth import HTTPBasicAuth
import json
import sys


def writeListToFile(list, fileName):
	with open('bulk_data/' + fileName, 'w') as fout:
		json.dump(list,  fout)

pageNumber = 1

while pageNumber < 11 :

	URL = "https://api.github.com/search/repositories?q=forks:>100&sort=forks&per_page=100&page=" + str(pageNumber)

	print(URL)

	r = requests.get(url = URL, auth=HTTPBasicAuth(sys.argv[1], sys.argv[2]))
	repositories = r.json()['items']

	writeListToFile(repositories, 'topForked1_' + str(pageNumber))

	pageNumber += 1



