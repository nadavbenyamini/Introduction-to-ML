from bs4 import BeautifulSoup
import requests

r = requests.get("https://govextra.gov.il/ministry-of-health/corona/corona-virus/spokesman-messages-corona/")
soup = BeautifulSoup(r.content, features="html.parser")
cards = soup.find_all("div", {"class": "card-body"})
cases = cards[0].find_all('p')

exposure_start = len(cases)
exposure_end = 0
for i in range(len(cases)):
    case = cases[i]
    if case.get_text()[::-1].find('מקומות חשיפה') > 0:
        exposure_start = i
    if case.get_text()[::-1].find('הנחיות לציבור') > 0:
        exposure_end = i
    if exposure_start < i < exposure_end:
        print(case.get_text()[::-1])
