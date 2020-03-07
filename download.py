#Put in pytorch/data/training_data/ShapeNetP2M/
import gdown
urls = [
    'https://drive.google.com/open?id=1dKM_6sR7SCdDQXX8dIQwtpaW3qRiA7pM',
    'https://drive.google.com/open?id=1hXq-kxEsDA0_DzcOFS0yzIvlKxwcgoGZ'
]

urls = [url.replace('open', 'uc') for url in urls]
for id, url in enumerate(urls):
    output = 'training_data.00{}'.format(id + 1)
    print('Donwloading {} ...'.format(output))
    gdown.download(url, output, quiet=False)

import subprocess
commands = ['mv training_data.001 pytorch/data/training_data/ShapeNetP2M/', 
            'mv training_data.002 pytorch/data/training_data/ShapeNetP2M/',
            'unzip training_data.001', 'unzip training_data.002']
for command in commands : 
	subprocess.check_call(command.split())
