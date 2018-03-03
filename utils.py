import time

def parse_time(time_string):
    result = time.mktime(time.strptime(time_string,"%d/%m/%Y %H:%M"))
    return int(result)

def format_time(utc_secs):
    return time.strftime("%d/%m/%Y %H:%M %Z",time.gmtime(utc_secs))