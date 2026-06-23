"""Upload changed files + compile live_bsd on remote Ubuntu 18.04."""
import paramiko
import os

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(
    os.environ.get('BSD_BUILD_HOST', '192.168.144.137'),
    username=os.environ.get('BSD_BUILD_USER', 'ubuntu'),
    password=os.environ.get('BSD_BUILD_PASSWORD'),
    timeout=30,
)

REMOTE_DIR = '/home/ubuntu/bsd_engine'
sftp = ssh.open_sftp()

# Upload changed files
files = [
    ('detect_engine/yolo_decode.c', True),
    ('detect_engine/live_bsd.c', True),
    ('detect_engine/test_npu_direct.c', True),
    ('detect_engine/detect_engine.c', True),
    ('detect_engine/detect_engine.h', True),
    ('detect_engine/awnn.h', True),
    ('detect_engine/preprocess.c', True),
    ('common/types.h', True),
    ('alarm_engine/alarm_engine.c', False),
    ('alarm_engine/alarm_engine.h', False),
    ('alarm_engine/zone_mgr.c', False),
    ('alarm_engine/tracker.c', False),
    ('detect_engine/Makefile', False),
]

for fname, required in files:
    local = rf'd:\WORK\CODE\BSD\deployment\board\{fname}'
    if os.path.exists(local):
        remote = f'{REMOTE_DIR}/{fname}'
        remote_dir = remote.rsplit('/', 1)[0]
        ssh.exec_command(f'mkdir -p "{remote_dir}"')[1].channel.recv_exit_status()
        sftp.put(local, remote)
        print(f'Uploaded: {os.path.basename(fname)}')
    elif required:
        print(f'MISSING: {local}')
        sftp.close()
        ssh.close()
        exit(1)

sftp.close()

# Compile
print('\n=== Compiling live_bsd ===')
stdin, stdout, stderr = ssh.exec_command(
    f'cd {REMOTE_DIR}/detect_engine && make clean && make live_bsd test_npu_direct 2>&1', timeout=120)
out = stdout.read().decode('utf-8', errors='replace')
err = stderr.read().decode('utf-8', errors='replace')
print(out)
for line in err.split('\n'):
    if any(kw in line for kw in ['error:', 'Error', 'undefined']):
        print(f'  !! {line[:300]}')

# Check binary
stdin, stdout, stderr = ssh.exec_command(
    f'file {REMOTE_DIR}/detect_engine/live_bsd {REMOTE_DIR}/detect_engine/test_npu_direct && '
    f'ls -lh {REMOTE_DIR}/detect_engine/live_bsd {REMOTE_DIR}/detect_engine/test_npu_direct', timeout=10)
check = stdout.read().decode().strip()
print(f'\n{check}')
if 'ELF' not in check:
    print('\n=== BUILD FAILED ===')
    stdin, stdout, stderr = ssh.exec_command(
        f'cd {REMOTE_DIR}/detect_engine && make live_bsd 2>&1', timeout=60)
    print(stdout.read().decode('utf-8', errors='replace'))
    print(stderr.read().decode('utf-8', errors='replace'))
    sftp = ssh.open_sftp()
    sftp.close()
    ssh.close()
    exit(1)

# Download
BIN_DIR = r'd:\WORK\CODE\BSD\artifacts\board_bin'
os.makedirs(BIN_DIR, exist_ok=True)
sftp = ssh.open_sftp()
sftp.get(f'{REMOTE_DIR}/detect_engine/live_bsd', f'{BIN_DIR}/live_bsd')
sz = os.path.getsize(f'{BIN_DIR}/live_bsd')
print(f'Downloaded live_bsd: {sz} bytes')
sftp.get(f'{REMOTE_DIR}/detect_engine/test_npu_direct', f'{BIN_DIR}/test_npu_direct')
sz = os.path.getsize(f'{BIN_DIR}/test_npu_direct')
print(f'Downloaded test_npu_direct: {sz} bytes')
sftp.close()
ssh.close()
print('\n=== Done ===')
