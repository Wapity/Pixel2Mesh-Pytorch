import requests


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


urls = [
    '1Qo1NGS0ByB9L9MbEQ2TdoYtw1pHqJR6u', '14OmDZt41W1KXQGk5qHuraQiXe1q7RghT',
    '1ElUDDpUPZULM4MGptuxJqMtUZxcGDLvJ', '1nvDDweOxTxFJ2m0p2KGrDaCi5wEeIDC7'
]

for id, file_id in enumerate(urls):
    output = 'training_data.7z.00{}'.format(id + 1)
    print('Donwloading {} ...'.format(output))
    download_file_from_google_drive(file_id, output)
