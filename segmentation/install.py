"""
Installe ce qu'il faut pour la segmentation et la lecture DICOM d'OMFS Planner, une seule fois par ordinateur.

Lancé par le bouton « Installer ce qui manque » de Blender (Préférences de l'extension, ou panneau Installation),
ou à la main :  python3 segmentation/install.py

Étapes (chacune sautée si déjà faite) :
  1. vérifie le Mac : puce Apple, macOS 14 ou plus récent (PyTorch 2.14 n'existe pas en dessous) ;
  2. uv, un petit installateur Python autonome (celui de Homebrew s'il existe, sinon téléchargé et vérifié) ;
  3. un environnement Python 3.12 dans segmentation/.venv (uv télécharge Python lui-même : rien à installer avant) ;
  4. les paquets aux versions exactes de requirements.lock.txt (PyTorch, nnU-Net, SimpleITK... ~1 Go) ;
  5. les poids du modèle DentalSegmentator (Dot et al., 2024, CC-BY-4.0 ; 230 Mo, somme SHA-256 vérifiée)
     dans segmentation/model/ ;
  6. un essai : les paquets s'importent, et le GPU de la puce Apple (MPS) est disponible.

Bibliothèque standard seulement (il tourne avec le Python de Blender). Rien n'est installé hors de ce dossier,
sauf le cache de uv (~/.cache/uv et ~/.local/share/uv : Python 3.12 et les paquets téléchargés).
Chaque ligne « ÉTAPE n/6 » et « AVANCE p » est lue par Blender pour la barre d'avancement ; la dernière ligne
est « TERMINÉ » ou « ÉCHEC : raison ».
"""

import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV = HERE / ".venv"
TOOLS = HERE / ".tools"
MODEL = HERE / "model"
LOCK = HERE / "requirements.lock.txt"
PYTHON_VERSION = "3.12"
MIN_MACOS = 14

UV_VERSION = "0.13.1"
UV_URL = f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/uv-aarch64-apple-darwin.tar.gz"
UV_SHA256 = "cfe454573412ff7380d15fc7dae5997ab53ffce72fce0a78e91ddbc84001acd2"

# Same file the DentalSegmentator extension of 3D Slicer downloads (identical to the Zenodo record 10829675).
WEIGHTS_URL = ("https://github.com/gaudot/SlicerDentalSegmentator/releases/download/v1.0.0-alpha/"
               "Dataset111_453CT_v100.zip")
WEIGHTS_SHA256 = "779fc3fffb4ee3caa371f7602520c5a254ad8d8618aead98e79d69e9874b7d35"
WEIGHTS_SIZE = 229698747
CHECKPOINT = MODEL / "Dataset111_453CT" / "nnUNetTrainer__nnUNetPlans__3d_fullres" / "fold_0" / "checkpoint_final.pth"

IMPORT_TEST = ("import torch, nnunetv2, SimpleITK, nrrd, scipy; "
               "print('MPS', torch.backends.mps.is_available())")
STEPS = 6


class Failure(RuntimeError):
    pass


def say(text):
    print(text, flush=True)


def step(n, text):
    say(f"ÉTAPE {n}/{STEPS} : {text}")


def run(command, **kwargs):
    """Run a command, its output passed through (Blender shows the last line)."""
    done = subprocess.run([str(c) for c in command], stdout=sys.stdout, stderr=subprocess.STDOUT, **kwargs)
    if done.returncode != 0:
        raise Failure(f"{Path(str(command[0])).name} a échoué (code {done.returncode})")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, target, size=None):
    """curl (always on macOS, uses the system's certificates), the progress read from the file's size."""
    process = subprocess.Popen(["curl", "-L", "--fail", "--silent", "--show-error", "-o", str(target), url],
                               stdout=sys.stdout, stderr=subprocess.STDOUT)
    while process.poll() is None:
        if size and Path(target).exists():
            say(f"AVANCE {min(99, int(100 * Path(target).stat().st_size / size))}")
        time.sleep(1.0)
    if process.returncode != 0:
        raise Failure(f"téléchargement impossible : {url} (connexion internet ?)")


# --- Steps ----------------------------------------------------------------------------------------

def check_mac():
    step(1, "vérification de l'ordinateur")
    if sys.platform != "darwin":
        raise Failure("macOS seulement pour l'instant")
    if platform.machine() != "arm64":
        raise Failure("puce Apple (M1 ou plus récente) nécessaire : PyTorch 2.14 n'existe pas pour Mac Intel")
    major = int(platform.mac_ver()[0].split(".")[0] or 0)
    if major < MIN_MACOS:
        raise Failure(f"macOS {MIN_MACOS} ou plus récent nécessaire (cet ordinateur : {platform.mac_ver()[0]})")
    say(f"macOS {platform.mac_ver()[0]}, puce Apple : ok")


def find_uv():
    for candidate in (TOOLS / "uv", shutil.which("uv"), "/opt/homebrew/bin/uv", Path.home() / ".local/bin/uv"):
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return Path(candidate)
    return None


def get_uv():
    step(2, "installateur Python (uv)")
    uv = find_uv()
    if uv is not None:
        say(f"uv déjà présent : {uv}")
        return uv
    TOOLS.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as work:
        archive = Path(work) / "uv.tar.gz"
        download(UV_URL, archive)
        if sha256(archive) != UV_SHA256:
            raise Failure("uv téléchargé corrompu (somme SHA-256 différente)")
        with tarfile.open(archive) as tar:
            member = next(m for m in tar.getmembers() if m.name.endswith("/uv") and m.isfile())
            member.name = "uv"
            try:
                tar.extract(member, TOOLS, filter="data")
            except TypeError:                          # Python before 3.12 (macOS's own python3 is 3.9)
                tar.extract(member, TOOLS)
    uv = TOOLS / "uv"
    uv.chmod(0o755)
    say(f"uv {UV_VERSION} installé : {uv}")
    return uv


def venv_python():
    return VENV / "bin" / "python"


def make_venv(uv):
    step(3, f"environnement Python {PYTHON_VERSION}")
    if venv_python().exists():
        say(f"déjà là : {VENV}")
        return
    run([uv, "venv", "--python", PYTHON_VERSION, VENV])


def install_packages(uv):
    step(4, "paquets (PyTorch, nnU-Net, SimpleITK… ~1 Go la première fois)")
    env = dict(os.environ, MACOSX_DEPLOYMENT_TARGET=f"{MIN_MACOS}.0")
    run([uv, "pip", "install", "--python", venv_python(), "-r", LOCK], env=env)


def get_weights():
    step(5, "poids du modèle DentalSegmentator (230 Mo)")
    if CHECKPOINT.exists():
        say(f"déjà là : {CHECKPOINT.parent.parent.parent}")
        return
    MODEL.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=MODEL) as work:      # same disk: no copy at the end
        archive = Path(work) / "weights.zip"
        download(WEIGHTS_URL, archive, WEIGHTS_SIZE)
        say("vérification de la somme SHA-256…")
        if sha256(archive) != WEIGHTS_SHA256:
            raise Failure("poids téléchargés corrompus (somme SHA-256 différente) : relancez l'installation")
        with zipfile.ZipFile(archive) as z:
            z.extractall(work)
        shutil.move(str(Path(work) / "Dataset111_453CT"), str(MODEL / "Dataset111_453CT"))
    say("poids installés")


def test():
    step(6, "essai")
    done = subprocess.run([str(venv_python()), "-c", IMPORT_TEST], capture_output=True, text=True)
    if done.returncode != 0:
        raise Failure("les paquets ne s'importent pas : " + (done.stderr.strip().splitlines() or ["?"])[-1])
    gpu = "GPU de la puce Apple disponible" if "MPS True" in done.stdout else "GPU indisponible : segmentation lente"
    say(f"paquets ok ; {gpu}")


def main():
    try:
        check_mac()
        uv = get_uv()
        make_venv(uv)
        install_packages(uv)
        get_weights()
        test()
    except Failure as error:
        say(f"ÉCHEC : {error}")
        return 1
    say("TERMINÉ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
