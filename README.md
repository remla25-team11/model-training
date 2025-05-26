# Model Training

This repository contains the code used to train our sentiment classification model.

The training logic is organized into modular scripts using a DVC pipeline:
- `prepare_data.py` – cleans and vectorizes the input
- `train_model.py` – trains and saves the classifier
- `evaluate_model.py` – evaluates performance and logs metrics

## Install Dependencies

```bash
pip install -r requirements.txt
```

## DVC Remote Setup (Google Drive via Custom Google Cloud Project)

This project uses Google Drive as a DVC remote to store and share datasets and trained models using a custom Google Cloud OAuth application.

![Screenshot from 2025-05-26 21-02-50](https://github.com/user-attachments/assets/02dc041d-14f1-4fdb-8905-09b1e46a539c)

### Setup Instructions

1. Install DVC with Google Drive support:

   ```bash
   pip install "dvc[gdrive]"
    ```

2. Place client_secrets.json (shared privately) into:

    ```bash
    .dvc/client_secrets.json
    ```

3. Modify .dvc/config

   replace 
   ``` bash
   gdrive_client_id = xxx.apps.googleusercontent.com
   gdrive_client_secret = xxx[core]
   ```
   with client_id and client_secret from the shared client_secrets.json file. 

5. Configure the remote

    ```bash
    dvc remote add -d myremote gdrive://1l7cRzuSYqVK5EAkAHJSMrVo5Nc2W8cbL
    dvc remote modify myremote gdrive_use_service_account false
    dvc remote modify myremote gdrive_client_id <your-client-id>
    dvc remote modify myremote gdrive_client_secret <your-client-secret>

    ```

6. Pull shared data and models:

    ```bash
    dvc pull
    ```

7. Push updates to remote:

    ```bash
    dvc push
    ```
## Tests
Tests run automatically, on every push or pull. Alternatively, they can be executed with:
```bash
pytest tests/
```
    
```
-- COVERAGE_REPORT --
Name                           Stmts   Miss  Cover
--------------------------------------------------
tests/test_development.py         30      0   100%
tests/test_infrastructure.py      22      2    91%
tests/test_monitoring.py          14      0   100%
tests/test_mutation.py            17      0   100%
tests/test_nondeterminism.py      12      0   100%
tests/test_performance.py         19      0   100%
tests/test_preprocess.py          45      0   100%
--------------------------------------------------
TOTAL                            159      2    99%
```
