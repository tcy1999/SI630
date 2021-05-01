import pandas as pd
import re

webpages = pd.read_csv('W21_webpages.csv')
Category = []
pattern1 = r"([-\w\.!#$%&'*+/=?`{|}~^]+)\s*@+\s*([a-zA-Z0-9\-]+)\s*\.\s*(org|net|com|gov)"
pattern2 = r"([-\w\.!#$%&'*+/=?`{|}~^]+)\s+[/\[]?[aA][tT][/\]]?\s+([a-zA-Z0-9\-]+)\s+[/\[]?[dD][oO][tT][/\]]?\s+(org|net|com|gov)"
for html in webpages['html']:
    result = re.search(pattern1, html)
    if not result:
        result = re.search(pattern2, html)
    if result:
        Category.append('{}@{}.{}'.format(result.group(1), result.group(2), result.group(3)))
    else:
        Category.append('None')
emails = pd.DataFrame(webpages['Id'])
emails['Category'] = Category
emails.to_csv('email-outputs.csv', index=False)
