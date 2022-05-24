#training and testing code

import aicrowd_helper
import train_submission_code


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


