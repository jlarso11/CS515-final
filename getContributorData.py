import json
import requests
from requests.auth import HTTPBasicAuth
import time
import sys


# https://stackabuse.com/reading-and-writing-json-to-a-file-in-python/
def readJsonFile(file_name):
	f = open(file_name)
	with f as json_file:
		return json.load(json_file)
	f.close()


def writeListToFile(list, fileName):
	f = open('dataToUse/' + fileName, 'w')
	with f as fout:
		json.dump(list,  fout)
	f.close()


def getContributors(name):
	url = 'https://api.github.com/repos/' + name + '/contributors'
	print(url)

	r = requests.get(url=url, auth=HTTPBasicAuth(sys.argv[1], sys.argv[2]))
	try:
		repo_contributors = r.json()
		r.close()

		if type(repo_contributors) is dict:
			print(repo_contributors)

		elif len(repo_contributors) > 1:

			return repo_contributors

	except ValueError:
		print('no contributors')

	return None


def getIssues(name):
	url = 'https://api.github.com/repos/' + name + '/issues?state=all'
	print(url)

	r = requests.get(url=url, auth=HTTPBasicAuth(sys.argv[1], sys.argv[2]))
	try:
		repo_issues = r.json()
		r.close()

		if type(repo_issues) is dict:
			print(repo_issues)

		else:
			pr_count = 0
			defect_count = 0
			for issue in repo_issues:
				if "pull_request" in issue.keys():
					pr_count += 1
				else:
					defect_count += 1
			return {
				"pull_requests": pr_count,
				"defects": defect_count
			}

	except ValueError:
		print('no issues')

	return None


repos = readJsonFile('dataToUse/topData_combined')

reposToSave = []

for repo in repos:
	contributors = getContributors(repo['full_name'])
	issues = getIssues(repo['full_name'])

	if issues is not None:
		repo['issues'] = issues

	if contributors is not None:
		repo['contributors'] = contributors
		reposToSave.append(repo)

	time.sleep(1)

writeListToFile(reposToSave, 'topData_complete_2')
