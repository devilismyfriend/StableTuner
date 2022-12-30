import filecmp
import importlib.util
import os
import shutil
import sys
import sysconfig
import subprocess
from pathlib import Path
import requests
import zipfile
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata

req_file = os.path.join(os.getcwd(), "requirements.txt")

def run(command, desc=None, errdesc=None, custom_env=None):
    if desc is not None:
        print(desc)

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, env=os.environ if custom_env is None else custom_env)

    if result.returncode != 0:

        message = f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}
stdout: {result.stdout.decode(encoding="utf8", errors="ignore") if len(result.stdout)>0 else '<empty>'}
stderr: {result.stderr.decode(encoding="utf8", errors="ignore") if len(result.stderr)>0 else '<empty>'}
"""
        raise RuntimeError(message)

    return result.stdout.decode(encoding="utf8", errors="ignore")

def check_versions():
    global req_file
    reqs = open(req_file, 'r')
    lines = reqs.readlines()
    reqs_dict = {}
    for line in lines:
        splits = line.split("==")
        if len(splits) == 2:
            key = splits[0]
            if "torch" not in key:
                if "diffusers" in key:
                    key = "diffusers"
                reqs_dict[key] = splits[1].replace("\n", "").strip()
    
    if os.name == "nt":
        reqs_dict["torch"] = "1.12.1+cu116"
        reqs_dict["torchvision"] = "0.13.1+cu116"

    checks = ["xformers","bitsandbytes", "diffusers", "transformers", "torch", "torchvision"]
    for check in checks:
        check_ver = "N/A"
        status = "[ ]"
        try:
            check_available = importlib.util.find_spec(check) is not None
            if check_available:
                check_ver = importlib_metadata.version(check)
                if check in reqs_dict:
                    req_version = reqs_dict[check]
                    if str(check_ver) == str(req_version):
                        status = "[+]"
                    else:
                        status = "[!]"
        except importlib_metadata.PackageNotFoundError:
            check_available = False
        if not check_available:
            status = "[!]"
            print(f"{status} {check} NOT installed.")
            if check == 'xformers':
                x_cmd = "-U -I --no-deps https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/f/xformers-0.0.14.dev0-cp310-cp310-win_amd64.whl"
                print(f"Installing xformers with: pip install {x_cmd}")
                run(f"pip install {x_cmd}", desc="Installing xformers")

        else:
            print(f"{status} {check} version {check_ver} installed.")


dreambooth_skip_install = os.environ.get('DREAMBOOTH_SKIP_INSTALL', False)

if not dreambooth_skip_install:
    check_versions()
    name = "StableTuner"    
    run(f'"{sys.executable}" -m pip install -r "{req_file}"', f"Checking {name} requirements...",
        f"Couldn't install {name} requirements.")

    # I think we only need to bump torch version to cu116 on Windows, as we're using prebuilt B&B Binaries...
    if os.name == "nt":
        torch_cmd = os.environ.get('TORCH_COMMAND', None)
        if torch_cmd is None:
            torch_cmd = 'pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url "https://download.pytorch.org/whl/cu116"'
                        
        run(f'"{sys.executable}" -m {torch_cmd}', "Checking/upgrading existing torch/torchvision installation", "Couldn't install torch")
        

        #if .cache directory in Path.home() exists
        hf_cache_dir = Path.home() / ".cache"
        if hf_cache_dir.exists():
            #check if huggingface exists
            hf_dir = hf_cache_dir / "huggingface"
            if hf_dir.exists():
                #check if accelerate exists
                accelerate_dir = hf_dir / "accelerate"
                if accelerate_dir.exists():
                    #print('test')
                    src_file = 'resources/accelerate_windows/accelerate_default_config.yaml'
                    dst_file = 'default_config.yaml'
                    #load from cwd
                    src = Path.cwd() / src_file
                    dst = accelerate_dir / dst_file
                    print(src)
                    if src.exists():
                        shutil.copy2(src, dst)
                        print(f"Updated {dst_file} in {accelerate_dir}")
        else:
            #make dirs
            hf_cache_dir.mkdir(parents=True, exist_ok=True)
            hf_dir = hf_cache_dir / "huggingface"
            hf_dir.mkdir(parents=True, exist_ok=True)
            accelerate_dir = hf_dir / "accelerate"
            accelerate_dir.mkdir(parents=True, exist_ok=True)
            src_file = 'accelerate_default_config.json'
            dst_file = 'default_config.json'
            src = Path.cwd() / src_file
            dst = accelerate_dir / dst_file
            if src.exists():
                if dst.exists():
                    shutil.copy2(src, dst)
                    print(f"Created {dst_file} in {accelerate_dir}")



base_dir = os.path.dirname(os.getcwd())
#repo = git.Repo(base_dir)
#revision = repo.rev_parse("HEAD")
#print(f"Dreambooth revision is {revision}")
check_versions()
# Check for "different" B&B Files and copy only if necessary
if os.name == "nt":
    python = sys.executable
    bnb_src = os.path.join(os.getcwd(), "resources/bitsandbytes_windows")
    bnb_dest = os.path.join(sysconfig.get_paths()["purelib"], "bitsandbytes")
    cudnn_src = os.path.join(os.getcwd(), "resources/cudnn_windows")
    #check if chudnn is in cwd
    if not os.path.exists(cudnn_src):
        print("Can't find CUDNN in resources, trying main folder...")
        cudnn_src = os.path.join(os.getcwd(), "cudnn_windows")
        if not os.path.exists(cudnn_src):
            cudnn_url = "https://b1.thefileditch.ch/mwxKTEtelILoIbMbruuM.zip"
            print(f"Downloading CUDNN 8.6")
            #download with requests
            r = requests.get(cudnn_url, allow_redirects=True)
            #save to cwd
            open('cudnn_windows.zip', 'wb').write(r.content)
            #unzip
            with zipfile.ZipFile('cudnn_windows.zip','r') as zip_ref:
                zip_ref.extractall(os.path.join(os.getcwd(),"resources/cudnn_windows"))
            #remove zip
            os.remove('cudnn_windows.zip')
            cudnn_src = os.path.join(os.getcwd(), "resources/cudnn_windows")

    cudnn_dest = os.path.join(sysconfig.get_paths()["purelib"], "torch", "lib")
    print(f"Checking for B&B files in {bnb_dest}")
    if not os.path.exists(bnb_dest):
        # make destination directory
        os.makedirs(bnb_dest, exist_ok=True)
    printed = False
    filecmp.clear_cache()
    for file in os.listdir(bnb_src):
        src_file = os.path.join(bnb_src, file)
        if file == "main.py":
            dest = os.path.join(bnb_dest, "cuda_setup")
            if not os.path.exists(dest):
                os.mkdir(dest)
        else:
            dest = bnb_dest
            if not os.path.exists(dest):
                os.mkdir(dest)
        dest_file = os.path.join(dest, file)
        status = shutil.copy2(src_file, dest)
    if status:
        print("Copied B&B files to destination")
    print(f"Checking for CUDNN files in {cudnn_dest}")
    if os.path.exists(cudnn_src):
        if os.path.exists(cudnn_dest):
            # check for different files
            filecmp.clear_cache()
            for file in os.listdir(cudnn_src):
                src_file = os.path.join(cudnn_src, file)
                dest_file = os.path.join(cudnn_dest, file)
                #if dest file exists, check if it's different
                if os.path.exists(dest_file):
                    status = shutil.copy2(src_file, cudnn_dest)
            if status:
                print("Copied CUDNN 8.6 files to destination")
    d_commit = '0ca1724'
    diffusers_cmd = f"git+https://github.com/huggingface/diffusers.git@{d_commit}#egg=diffusers --force-reinstall"
    run(f'"{python}" -m pip install {diffusers_cmd}', f"Installing Diffusers {d_commit} commit", "Couldn't install diffusers")
    #install requirements file
    t_commit = '491a33d'
    trasn_cmd = f"git+https://github.com/huggingface/transformers.git@{t_commit}#egg=transformers --force-reinstall"
    run(f'"{python}" -m pip install {trasn_cmd}', f"Installing Transformers {t_commit} commit", "Couldn't install transformers")

    req_file = os.path.join(os.getcwd(), "requirements.txt")
    run(f'"{python}" -m pip install -r "{req_file}"', "Updating requirements", "Couldn't install requirements")
    