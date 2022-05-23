import minerl
import minerl.data
import os

if __name__ == "__main__":
    data_dir = os.getenv('MINERL_DATA_ROOT', 'data')
    data_dir = 'data' if not data_dir else data_dir
    BASALT_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLBasaltFindCave-v0')

    print("Verifying (and downloading) MineRL dataset..\n"
          "\t\tmake sure your MINERL_DATA_ROOT is set.\n\n")

    print("Data directory is {}".format(data_dir))
    should_download = True
    try:
        data = minerl.data.make(BASALT_GYM_ENV, data_dir=data_dir)
        assert len(data._get_all_valid_recordings(data.data_dir)) > 0
        should_download = False
    except FileNotFoundError:
        print("The data directory does not exist!")
    except RuntimeError:
        print("The data contained in your data directory is out of date! data_dir={}".format(data_dir))
    except AssertionError:
        print(f"No {BASALT_GYM_ENV} data found. Did the data really download correctly?" )

    if should_download:
        print(" Downloading the dataset...")
        minerl.data.download(directory=data_dir, competition="basalt")

    print("Dataset downloaded!")
