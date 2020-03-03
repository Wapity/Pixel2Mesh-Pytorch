import gdown
urls = [
    'https://drive.google.com/open?id=1Qo1NGS0ByB9L9MbEQ2TdoYtw1pHqJR6u&export=download',
    'https://drive.google.com/open?id=14OmDZt41W1KXQGk5qHuraQiXe1q7RghT&export=download',
    'https://drive.google.com/uc?id=1ElUDDpUPZULM4MGptuxJqMtUZxcGDLvJ&export=download',
    'https://drive.google.com/open?id=1nvDDweOxTxFJ2m0p2KGrDaCi5wEeIDC7&export=download'
]

urls = [url.replace('open', 'uc') for url in urls]
for id, url in enumerate(urls):
    output = 'training_data.7z.00{}'.format(id + 1)
    print('Donwloading {} ...'.format(output))
    gdown.download(url, output, quiet=False)
