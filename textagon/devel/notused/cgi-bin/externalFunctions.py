### PID Termination Function ###
# https://stackoverflow.com/questions/4789837/how-to-terminate-a-python-subprocess-launched-with-shell-true
def kill(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive = True):
        proc.kill()
    process.kill()

### Chunking Function ###
# http://stackoverflow.com/questions/2130016/splitting-a-list-of-arbitrary-size-into-only-roughly-n-equal-parts
def chunks(l, amount):
    if amount < 1:
        raise ValueError('amount must be positive integer')
    chunk_len = len(l) // amount
    leap_parts = len(l) % amount
    remainder = amount // 2  # make it symmetrical
    i = 0
    while i < len(l):
        remainder += leap_parts
        end_index = i + chunk_len
        if remainder >= amount:
            remainder -= amount
            end_index += 1
        yield l[i:end_index]
        i = end_index