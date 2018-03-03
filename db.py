import pymysql
import os

from retrying import retry


@retry(stop_max_attempt_number=3, wait_fixed=2000)
def get_master_connection():
    if 'ENV_SILENCE' not in os.environ or os.environ['ENV_SILENCE'] == 0:
        print('Initiating master db connection...')
    conn = pymysql.connect(
        host=config['SMAPPEEMASTER']['host'],
        user=config['SMAPPEEMASTER']['user'],
        passwd=config['SMAPPEEMASTER']['pass'],
        db=config['SMAPPEEMASTER']['db']
    )
    return conn


@retry(stop_max_attempt_number=3, wait_fixed=2000)
def get_dev_connection():
    if 'ENV_SILENCE' not in os.environ or os.environ['ENV_SILENCE'] == 0:
        print('Initiating dev db connection...')
    conn = pymysql.connect(
        host=config['SMAPPEEDEV']['host'],
        user=config['SMAPPEEDEV']['user'],
        passwd=config['SMAPPEEDEV']['pass'],
        db=config['SMAPPEEDEV']['db']
    )
    return conn


@retry(stop_max_attempt_number=3, wait_fixed=2000)
def get_data_connection():
    if 'ENV_SILENCE' not in os.environ or os.environ['ENV_SILENCE'] == 0:
        print('Initiating data db connection...')
    conn = pymysql.connect(
        host=config['SMAPPEEDATA']['host'],
        user=config['SMAPPEEDATA']['user'],
        passwd=config['SMAPPEEDATA']['pass'],
        db=config['SMAPPEEDATA']['db']
    )
    return conn


@retry(stop_max_attempt_number=3, wait_fixed=2000)
def get_data_alchemy_connection():
    if 'ENV_SILENCE' not in os.environ or os.environ['ENV_SILENCE'] == 0:
        print('Initiating data sql alchemy db connection...')
    conn = create_engine('mysql+pymysql://%s:%s@%s:%s/%s' % (
        config['SMAPPEEDATA']['user'],
        config['SMAPPEEDATA']['pass'],
        config['SMAPPEEDATA']['host'],
        config['SMAPPEEDATA']['port'],
        config['SMAPPEEDATA']['db']),
        echo=False
    )
    return conn