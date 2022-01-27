import json

# Script used to gather all the markdown cells in the jupyter notebook, to be able to use Antidote to correct wrongs in sentences. 
# I'm used to do only few mistakes, but in to be sure that the project is really clean !

f = open("projet.ipynb", "r", encoding='utf-8')
text_json = json.loads(f.read())
f.close()

markdown_cells = [cell for cell in text_json['cells'] if cell['cell_type'] == "markdown"]

f = open("report.md", "wb")
for md in markdown_cells:
	for line in md['source']:
		f.write(line.encode('utf8'))
	f.write('\n\n'.encode('utf8'))
	f.write("------------------------------------------------------------------------------".encode('utf8'))
	f.write('\n\n'.encode('utf8'))
f.close()