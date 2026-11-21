# OMFS Planner

Planification chirurgicale virtuelle et conception de guides chirurgicaux imprimés en 3D pour la chirurgie
buccale et maxillo-faciale (génioplastie, Le Fort I, BSSO, occlusion segmentaire), dans Blender. Gratuit.

> Outil de recherche et d'enseignement, **pas un dispositif médical homologué**. Chaque plan et chaque guide doivent
> être vérifiés par le chirurgien.

Démarrage d'un cas : onglet **OMFS** > **Case** > **Nouveau cas**. Il suffit d'un CBCT (dossier DICOM) et des deux
empreintes optiques (STL). La segmentation IA tourne en arrière-plan, puis les étapes suivantes sont cochées une à
une dans le panneau. Le déroulement clinique complet est dans [docs/workflow.md](docs/workflow.md).

## Installation en une ligne (macOS 14 ou plus récent, puce Apple)

Ouvrez le Terminal (Applications > Utilitaires) et collez :

```sh
curl -fsSL https://raw.githubusercontent.com/thomaslgiguere-tech/omfs-planner/main/tools/install_mac.sh | sh
```

Le script installe tout ce qui manque, en sautant ce qui est déjà là ([tools/install_mac.sh](tools/install_mac.sh)) :
- **Blender 5.2** : téléchargé sur blender.org, somme de contrôle officielle vérifiée. Une autre version de
  Blender déjà installée n'est pas touchée.
- **Outils Xcode d'Apple** (git) : une fenêtre d'Apple s'ouvre, cliquez Installer.
- **Le projet**, dans `~/Projects/omfs-planner`.
- **L'extension**, reliée à Blender et activée.
- **La segmentation** : environnement Python (~1 Go) et poids du modèle.

Comptez 15 à 30 min selon la connexion. Relancer la même ligne plus tard sert aussi de mise à jour.
Ensuite : ouvrez Blender, onglet **OMFS** (touche N dans la vue 3D) > **Case** > **Nouveau cas**.

## Installation pas à pas (macOS 14 ou plus récent, puce Apple)

1. **Blender 5.2 LTS** : <https://www.blender.org/download/>
2. **Le projet**, dans le Terminal (la première commande `git` propose d'installer les outils Xcode d'Apple :
   acceptez) :
   ```sh
   git clone https://github.com/thomaslgiguere-tech/omfs-planner.git ~/Projects/omfs-planner
   ~/Projects/omfs-planner/tools/dev_link.sh 5.2
   ```
   `dev_link.sh` relie l'extension de ce dossier à Blender : Blender lit toujours la version présente ici.
3. Dans Blender : Édition > Préférences > Extensions, activer **OMFS Planner**.
4. Onglet **OMFS** > **Installation et mises à jour** (aussi dans les préférences de l'extension) : une ligne
   ✓ / ✗ par composant. Cliquez **Installer ce qui manque** : l'environnement Python de la segmentation (~1 Go)
   et les poids du modèle (230 Mo) sont téléchargés en arrière-plan, une seule fois, et vérifiés. Blender reste
   utilisable pendant ce temps (10-20 min selon la connexion).
5. Pour l'onglet Impression seulement : PrusaSlicer et UVtools (liens « Télécharger » dans le même panneau).

La planification elle-même n'a besoin que de Blender. Sans la segmentation, on importe des STL déjà segmentés
(Case > Autres imports).

Ce que fait l'installation (`segmentation/install.py`, relançable aussi à la main avec `python3
segmentation/install.py`) :
- elle télécharge `uv`, un installateur Python autonome, dans `segmentation/.tools/` ;
- elle crée un Python 3.12 dans `segmentation/.venv/`, avec les versions exactes de
  `segmentation/requirements.lock.txt` ;
- elle place les poids de DentalSegmentator (Dot et al., J Dent 2024, licence CC-BY-4.0) dans
  `segmentation/model/`.

Rien n'est installé ailleurs, sauf le cache de uv dans le dossier personnel.

## Mises à jour

Panneau **Installation et mises à jour** > **Rechercher**, puis **Mettre à jour**, puis redémarrez Blender. C'est un
`git pull` : il ne remplace jamais une modification locale. Vos cas ne sont jamais touchés : ils ne sont pas
dans ce dossier.

## Données des patients

Aucune donnée de patient n'entre dans ce dépôt. Le `.gitignore` bloque les maillages, DICOM, volumes, `.blend`
et les dossiers `cases/` et `data/`. Les segmentations sont écrites à côté du CBCT, sur votre ordinateur.

## Développement

- Tests : `/Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup --python tests/run_tests.py`
- Architecture, règles et conventions : [CLAUDE.md](CLAUDE.md). Termes Blender ↔ Meshmixer : [docs/GLOSSARY.md](docs/GLOSSARY.md).

Licence : GPL-3.0-or-later (voir `blender_manifest.toml`).
