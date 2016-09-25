import pandas as pd
import numpy as np
import urllib.request as urllib2

name = 'VABB'
base_url_1 = 'https://www.wunderground.com/history/airport/'
base_url_2 = '/24/MonthlyHistory.html?format=1'

years = range(1996, 2006)
months = range(1, 13)

lines = []
header = None

for year in years:
    for month in months:
        print('Scraping', year, month)

        base_url = base_url_1 + name + '/' + str(year) + '/' + str(month) + base_url_2
        response = urllib2.urlopen(base_url)
        html = response.read()

        csv = str(html).replace('<br />', '').replace('\\n', '\n')

        ls = csv.split('\n')
        ls = ls[1:-1]
        header = ls.pop(0)

        lines.extend(ls)

print('Writing ...')
f = open(name + '.csv', 'w')
f.write(header + '\n')
for line in lines:
    f.write(line + '\n')
f.close()
