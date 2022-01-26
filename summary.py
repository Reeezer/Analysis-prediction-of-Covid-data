import json

# Script used to gather all the markdown cells in the jupyter notebook, to make a summary using all the titles.

f = open("projet.ipynb", "r", encoding='utf-8')
text_json = json.loads(f.read())
f.close()

numbers = [f"{i}" for i in range(100)]

markdown_cells = [cell for cell in text_json['cells'] if cell['cell_type'] == "markdown"]

f = open("summary.md", "wb")
for md in markdown_cells:
    for line in md['source']:
        if '#' in line:
            title = line.rstrip('\n')
            
            i = 0
            while title[i] in '# ':
                i += 1
                
            j = i
            tabs = ''       
            while title[j] in numbers or title[j] in '.':
                if title[j] in '1234567890':
                    tabs += '\t'
                j += 1
            
            number = ''
            if title[j-1] == '.':
                number = title[j-2]
            else:
                number = title[j-1]
            number += '.'
            
            title = title[j+1:]
            hashtag = title.replace(' ', '-')
            hashtag = hashtag.replace('é', 'e')
            hashtag = hashtag.replace('è', 'e')
            hashtag = hashtag.replace('ê', 'e')
            
            # summary = f"{tabs[1:]}{number} [{title}](#{hashtag})"
            summary = f"{tabs[1:]}{number} {title}"
            
            f.write(summary.encode('utf8'))
            f.write('\n'.encode('utf8'))
f.close()