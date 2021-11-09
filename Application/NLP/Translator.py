from easynmt import EasyNMT
model = EasyNMT('opus-mt')

postagged =   [('Good', 'PRP'), ('Morning', 'VBP')]

postagged = (postagged[0][0])
translation = model.translate(postagged, source_lang='en', target_lang='tl')
print(translation)