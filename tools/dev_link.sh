#!/bin/sh
# Symlink the extension and the app template from this repo into the Blender user
# directories, so Blender loads them straight from the working copy (development mode).
# Usage: tools/dev_link.sh [blender_version]   (default 5.2)
set -e
VERSION="${1:-5.2}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
BLENDER_USER="$HOME/Library/Application Support/Blender/$VERSION"

link() {  # link <source> <destination>
    mkdir -p "$(dirname "$2")"
    if [ -e "$2" ] && [ ! -L "$2" ]; then
        echo "Refusing to replace real folder: $2 (move it away first)" >&2
        exit 1
    fi
    ln -sfn "$1" "$2"
    echo "$2 -> $1"
}

link "$REPO/extension/omfs_planner" "$BLENDER_USER/extensions/user_default/omfs_planner"
link "$REPO/app_template/omfs_planner" "$BLENDER_USER/scripts/startup/bl_app_templates_user/omfs_planner"
