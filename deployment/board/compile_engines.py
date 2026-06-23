"""Cross-compile BSD detect_engine + alarm_engine + main for V853 board."""
import paramiko
import os

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(
    os.environ.get('BSD_BUILD_HOST', '192.168.144.136'),
    username=os.environ.get('BSD_BUILD_USER', 'ubuntu'),
    password=os.environ.get('BSD_BUILD_PASSWORD'),
    timeout=30,
)

REMOTE_DIR = '/home/ubuntu/bsd_engine'
ssh.exec_command(f'mkdir -p {REMOTE_DIR}/detect_engine {REMOTE_DIR}/alarm_engine {REMOTE_DIR}/common')

sftp = ssh.open_sftp()

# Upload common header
sftp.put(r'd:\WORK\CODE\BSD\deployment\board\common\types.h', f'{REMOTE_DIR}/common/types.h')

# Upload detect_engine files
for f in ['detect_engine.c', 'detect_engine.h', 'preprocess.c', 'yolo_decode.c', 'main.c', 'Makefile']:
    local = rf'd:\WORK\CODE\BSD\deployment\board\detect_engine\{f}'
    sftp.put(local, f'{REMOTE_DIR}/detect_engine/{f}')

# Upload alarm_engine files
for f in ['alarm_engine.c', 'alarm_engine.h', 'zone_mgr.c', 'tracker.c']:
    local = rf'd:\WORK\CODE\BSD\deployment\board\alarm_engine\{f}'
    sftp.put(local, f'{REMOTE_DIR}/alarm_engine/{f}')

print('Files uploaded')

# Compile
print('\n=== Compiling BSD engines ===')
stdin, stdout, stderr = ssh.exec_command(
    f'cd {REMOTE_DIR}/detect_engine && make clean && make 2>&1', timeout=120)
out = stdout.read().decode('utf-8', errors='replace')
err = stderr.read().decode('utf-8', errors='replace')
print(out)
if err:
    for line in err.split('\n'):
        if any(kw in line for kw in ['error:', 'Error', 'undefined']):
            print(f'  !! {line[:200]}')

# Check results
for fname in ['libdetect_engine.so', 'bsd_detect']:
    stdin, stdout, stderr = ssh.exec_command(
        f'file {REMOTE_DIR}/detect_engine/{fname} && ls -lh {REMOTE_DIR}/detect_engine/{fname}', timeout=10)
    print(f'  {stdout.read().decode().strip()}')

# Verify all expected binaries exist
stdin, stdout, stderr = ssh.exec_command(
    f'test -f {REMOTE_DIR}/detect_engine/bsd_detect && echo "bsd_detect OK" || echo "bsd_detect MISSING"', timeout=10)
has_bin = 'OK' in stdout.read().decode()
stdin, stdout, stderr = ssh.exec_command(
    f'test -f {REMOTE_DIR}/detect_engine/libdetect_engine.so && echo "libdetect_engine.so OK" || echo "libdetect_engine.so MISSING"', timeout=10)
has_so = 'OK' in stdout.read().decode()

if not has_bin:
    print('\n=== BUILD FAILED — showing full error output ===')
    stdin, stdout, stderr = ssh.exec_command(
        f'cd {REMOTE_DIR}/detect_engine && make 2>&1', timeout=60)
    print(stdout.read().decode('utf-8', errors='replace'))
    print(stderr.read().decode('utf-8', errors='replace'))
    sftp.close()
    ssh.close()
    exit(1)

# Download binaries
BIN_DIR = r'd:\WORK\CODE\BSD\artifacts\board_bin'
os.makedirs(BIN_DIR, exist_ok=True)

for fname in ['libdetect_engine.so', 'bsd_detect']:
    sftp.get(f'{REMOTE_DIR}/detect_engine/{fname}', f'{BIN_DIR}/{fname}')
    sz = os.path.getsize(f'{BIN_DIR}/{fname}')
    print(f'  Downloaded {fname}: {sz} bytes')

sftp.close()
ssh.close()
print(f'\n=== Done. Binaries in {BIN_DIR} ===')
