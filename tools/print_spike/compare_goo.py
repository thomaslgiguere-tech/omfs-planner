"""Compare the print settings of a .goo with a reference .goo (e.g. one exported by Lychee for the same printer
and resin). Reads the headers only (UVtoolsCmd print-properties), never the layers.

python3 tools/print_spike/compare_goo.py reference.goo candidate.goo
Exit code 0 if every setting matches, 1 otherwise."""
import subprocess
import sys

UVTOOLS = "/Applications/UVtools.app/Contents/MacOS/UVtoolsCmd"
SETTINGS = """Version ResolutionX ResolutionY DisplayWidth DisplayHeight MachineZ DisplayMirror AntiAliasing LayerHeight
BottomLayerCount TransitionLayerType TransitionLayerCount BottomExposureTime ExposureTime BottomLightOffDelay
LightOffDelay BottomWaitTimeBeforeCure WaitTimeBeforeCure BottomWaitTimeAfterCure WaitTimeAfterCure
BottomLiftHeight BottomLiftSpeed LiftHeight LiftSpeed BottomLiftHeight2 BottomLiftSpeed2 LiftHeight2 LiftSpeed2
BottomWaitTimeAfterLift WaitTimeAfterLift BottomRetractSpeed RetractSpeed BottomRetractHeight2 BottomRetractSpeed2
RetractHeight2 RetractSpeed2 BottomLightPWM LightPWM MachineName""".split()


def settings(path):
    out = subprocess.run([UVTOOLS, "print-properties", path], capture_output=True, text=True).stdout
    found = {}
    for line in out.splitlines():
        key, sep, value = line.partition(": ")
        if sep and key in SETTINGS and key not in found:
            found[key] = value.strip()
    return found


if __name__ == "__main__":
    ref, new = settings(sys.argv[1]), settings(sys.argv[2])
    differ = [k for k in SETTINGS if ref.get(k) != new.get(k)]
    for k in SETTINGS:
        mark = "DIFF" if k in differ else "ok  "
        print(f"{mark} {k:24} reference {ref.get(k, '-'):>22}   ours {new.get(k, '-'):>22}")
    print(f"{len(SETTINGS) - len(differ)}/{len(SETTINGS)} settings match")
    sys.exit(1 if differ else 0)
