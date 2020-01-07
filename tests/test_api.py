import requests
import json
import pytest

test_data = json.dumps(
    {
        'data': [
            {
                "age": 40,
                "bmi": 22.2,
                "diab_pred": 2.1,
                "diastolic_bp": 100,
                "glucose_conc": 1,
                "insulin": 1,
                "num_preg": 1,
                "thickness": 1
            }
        ]
    }
)
test_data = str(test_data)

def test_api(score_url, score_key):
    assert score_url != None

    if score_key is None:
        headers = {'Content-Type':'application/json'}
    else:
        headers = {'Content-Type':'application/json', 'Authorization':('Bearer ' + score_key)}

    resp = requests.post(score_url, test_data, headers=headers)
    assert resp.status_code == requests.codes.ok
    assert resp.text != None
    assert resp.headers.get('content-type') == 'application/json'
    assert int(resp.headers.get('Content-Length')) > 0