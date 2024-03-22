import subprocess


def start(executable_file):
    return subprocess.Popen(
        executable_file,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)


def read(process):
    process.stdin.flush()
    res = process.stdout.readline().decode("utf-8").strip()
    return res


def write(process, message):
    process.stdin.write(f"{message.strip()}\n".encode("utf-8"))
    process.stdin.flush()


def terminate(process):
    process.stdin.close()
    process.terminate()
    process.wait(timeout=1)


# process = start("./dummy.py")
# write(process, "hello dummy")
# print(read(process))
# terminate(process)