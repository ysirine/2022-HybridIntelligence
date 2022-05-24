#Having a CUDA error in this code still hasn't solved it

import aicrowd_helper
import train_submission_code
import testKairos

import os
EVALUATION_RUNNING_ON = os.getenv('EVALUATION_RUNNING_ON', None)
EVALUATION_STAGE='testing'
EXITED_SIGNAL_PATH = os.getenv('EXITED_SIGNAL_PATH', 'shared/exited')

if EVALUATION_STAGE in ['all', 'testing']:
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
