#training and testing code

import aicrowd_helper
import train_submission_code
import testKairos

import os
EVALUATION_RUNNING_ON='local'
EVALUATION_STAGE='training'
EXITED_SIGNAL_PATH='shared/training_exited'
ENABLE_AICROWD_JSON_OUTPUT='False'

# Training Phase
if EVALUATION_STAGE in ['training']:
    aicrowd_helper.training_start()
    try:
        train_submission_code.main()
        aicrowd_helper.training_end()
    except Exception as e:
        aicrowd_helper.training_error()
        print(e)

EVALUATION_STAGE='testing'
EXITED_SIGNAL_PATH='shared/exited'
# Testing Phase
if EVALUATION_STAGE in ['testing']:
    if EVALUATION_RUNNING_ON in ['local']:
        try:
            os.remove(EXITED_SIGNAL_PATH)
        except FileNotFoundError:
            pass
    aicrowd_helper.inference_start()
    try:
        testKairos.main()
        aicrowd_helper.inference_end()
    except Exception as e:
        aicrowd_helper.inference_error()
        print(e)
    if EVALUATION_RUNNING_ON in ['local']:
        from pathlib import Path
        Path(EXITED_SIGNAL_PATH).touch()

