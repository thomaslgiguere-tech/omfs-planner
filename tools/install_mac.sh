#!/bin/sh
# OMFS Planner, installation complète sur un Mac (puce Apple, macOS 14+), en une ligne de Terminal :
#
#   curl -fsSL https://raw.githubusercontent.com/thomaslgiguere-tech/omfs-planner/main/tools/install_mac.sh | sh
#
# Chaque étape est sautée si elle est déjà faite ; relancer le script sert aussi de mise à jour.
#   1. vérifie le Mac ;
#   2. Blender 5.2 (blender.org, somme SHA-256 officielle vérifiée) dans Applications, s'il n'y est pas déjà ;
#   3. git (outils Xcode d'Apple : une fenêtre d'Apple demande de les installer, le script attend) ;
#   4. le projet, cloné dans ~/Projects/omfs-planner (ou mis à jour s'il y est déjà) ;
#   5. l'extension reliée à Blender et activée ;
#   6. la segmentation : environnement Python (~1 Go) et poids du modèle (segmentation/install.py).
#
# Variables facultatives : OMFS_DIR (dossier du projet), OMFS_NO_SEGMENTATION=1 (sauter l'étape 6).
set -e

BLENDER_VERSION="5.2.2"
BLENDER_SERIES="5.2"
REPO_URL="https://github.com/thomaslgiguere-tech/omfs-planner.git"
DIR="${OMFS_DIR:-$HOME/Projects/omfs-planner}"
RELEASE="https://download.blender.org/release/Blender$BLENDER_SERIES"
DMG="blender-$BLENDER_VERSION-macos-arm64.dmg"

say()  { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
fail() { printf '\n\033[31mÉCHEC : %s\033[0m\n' "$1" >&2; exit 1; }

# --- 1. Le Mac ----------------------------------------------------------------------------------
say "1/6 Vérification du Mac"
[ "$(uname -s)" = "Darwin" ] || fail "macOS seulement"
[ "$(uname -m)" = "arm64" ] || fail "puce Apple (M1 ou plus récente) nécessaire"
MACOS_MAJOR=$(sw_vers -productVersion | cut -d. -f1)
[ "$MACOS_MAJOR" -ge 14 ] || fail "macOS 14 ou plus récent nécessaire (ce Mac : $(sw_vers -productVersion))"
echo "macOS $(sw_vers -productVersion), puce Apple : ok"

# --- 2. Blender 5.2 -----------------------------------------------------------------------------
say "2/6 Blender $BLENDER_SERIES"
version_of() { defaults read "$1/Contents/Info.plist" CFBundleShortVersionString 2>/dev/null || echo ""; }
BLENDER_APP=""
for app in "/Applications/Blender.app" "/Applications/Blender $BLENDER_SERIES.app"; do
    case "$(version_of "$app")" in "$BLENDER_SERIES"*) BLENDER_APP="$app"; break ;; esac
done
if [ -n "$BLENDER_APP" ]; then
    echo "déjà installé : $BLENDER_APP ($(version_of "$BLENDER_APP"))"
else
    # Another Blender version stays untouched: this one goes next to it.
    if [ -e "/Applications/Blender.app" ]; then BLENDER_APP="/Applications/Blender $BLENDER_SERIES.app"
    else BLENDER_APP="/Applications/Blender.app"; fi
    WORK=$(mktemp -d)
    trap 'hdiutil detach -quiet "$WORK/mount" 2>/dev/null || true; rm -rf "$WORK"' EXIT
    echo "téléchargement de Blender $BLENDER_VERSION (~400 Mo)…"
    curl -fL --progress-bar -o "$WORK/$DMG" "$RELEASE/$DMG" || fail "téléchargement de Blender impossible"
    EXPECTED=$(curl -fsSL "$RELEASE/blender-$BLENDER_VERSION.sha256" | awk -v f="$DMG" '$2 == f {print $1}')
    [ -n "$EXPECTED" ] || fail "somme de contrôle officielle de Blender introuvable"
    [ "$(shasum -a 256 "$WORK/$DMG" | awk '{print $1}')" = "$EXPECTED" ] || fail "Blender téléchargé corrompu (somme SHA-256 différente)"
    echo "somme SHA-256 vérifiée"
    mkdir "$WORK/mount"
    hdiutil attach -quiet -nobrowse -readonly -mountpoint "$WORK/mount" "$WORK/$DMG"
    ditto "$WORK/mount/Blender.app" "$BLENDER_APP" || fail "copie dans Applications impossible"
    hdiutil detach -quiet "$WORK/mount"
    echo "installé : $BLENDER_APP"
fi
BLENDER="$BLENDER_APP/Contents/MacOS/Blender"
BLENDER_PYTHON=$(ls "$BLENDER_APP"/Contents/Resources/"$BLENDER_SERIES"/python/bin/python3.* 2>/dev/null | head -1)

# --- 3. git ---------------------------------------------------------------------------------------
say "3/6 git (outils Xcode d'Apple)"
if ! xcode-select -p >/dev/null 2>&1; then
    echo "Une fenêtre d'Apple propose d'installer les outils de développement : cliquez Installer."
    xcode-select --install >/dev/null 2>&1 || true
    printf "en attente de la fin de l'installation"
    i=0
    until xcode-select -p >/dev/null 2>&1; do
        sleep 10; printf "."; i=$((i + 1))
        [ $i -lt 180 ] || fail "outils Xcode toujours absents après 30 min : relancez le script une fois installés"
    done
    echo
fi
echo "git $(git --version | awk '{print $3}') : ok"

# --- 4. Le projet ---------------------------------------------------------------------------------
say "4/6 Le projet"
if [ -d "$DIR/.git" ]; then
    echo "déjà là : $DIR ; mise à jour"
    git -C "$DIR" pull --ff-only || fail "mise à jour refusée : des fichiers ont été modifiés dans $DIR"
else
    [ ! -e "$DIR" ] || fail "$DIR existe déjà et n'est pas un clone du projet (OMFS_DIR=autre/dossier pour choisir)"
    mkdir -p "$(dirname "$DIR")"
    git clone "$REPO_URL" "$DIR"
fi

# --- 5. L'extension dans Blender -----------------------------------------------------------------
say "5/6 Extension reliée à Blender et activée"
sh "$DIR/tools/dev_link.sh" "$BLENDER_SERIES"
# Enable it and save the preferences (Blender in the background, no window).
"$BLENDER" -b --python-expr "
import bpy, addon_utils
name = 'bl_ext.user_default.omfs_planner'
addon_utils.enable(name, default_set=True, persistent=True)
bpy.ops.wm.save_userpref()
print('OMFS_ENABLED', name in bpy.context.preferences.addons)
" 2>&1 | grep -q "OMFS_ENABLED True" && echo "extension activée" \
    || echo "à faire à la main : Blender > Édition > Préférences > Extensions > activer OMFS Planner"

# --- 6. La segmentation ---------------------------------------------------------------------------
say "6/6 Segmentation (environnement Python ~1 Go + poids du modèle 230 Mo)"
if [ "${OMFS_NO_SEGMENTATION:-0}" = "1" ]; then
    echo "sautée (OMFS_NO_SEGMENTATION=1) : bouton « Installer ce qui manque » dans Blender plus tard"
elif [ -z "$BLENDER_PYTHON" ]; then
    echo "Python de Blender introuvable : bouton « Installer ce qui manque » dans Blender"
else
    "$BLENDER_PYTHON" -I "$DIR/segmentation/install.py" | grep -v '^AVANCE' \
        || echo "Segmentation non installée : relancez depuis Blender (Installation et mises à jour)"
fi

say "Terminé"
echo "Ouvrez Blender ($BLENDER_APP), onglet OMFS (touche N dans la vue 3D) > Case > Nouveau cas."
echo "Mises à jour : panneau Installation et mises à jour > Rechercher, ou relancez cette même ligne."
