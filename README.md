# OMFS Planner

Planification chirurgicale virtuelle et conception de guides chirurgicaux imprimés en 3D pour la chirurgie
buccale et maxillo-faciale (génioplastie, Le Fort I, BSSO, occlusion segmentaire), dans Blender. Gratuit.

> Outil de recherche et d'enseignement, **pas un dispositif médical homologué**. Chaque plan et chaque guide doivent
> être vérifiés par le chirurgien.

Démarrage d'un cas : onglet **OMFS** > **Case** > **Nouveau cas**. Il suffit d'un CBCT (dossier DICOM) et des deux
empreintes optiques (STL). La segmentation IA tourne en arrière-plan, puis les étapes suivantes sont cochées une à
une dans le panneau. Le déroulement clinique complet est dans [docs/workflow.md](docs/workflow.md).

## Installation (macOS, puce Apple)

1. **Blender 5.2 LTS** : <https://www.blender.org/download/>
2. **Le projet**, avec git (Terminal) :
   ```sh
   git clone https://github.com/<compte>/omfs-planner.git ~/Projects/omfs-planner
   cd ~/Projects/omfs-planner
   tools/dev_link.sh 5.2
   ```
   `dev_link.sh` relie l'extension et le modèle d'application de ce dossier à Blender (liens symboliques).
   Blender lit donc toujours la version présente dans ce dossier.
3. Dans Blender : Édition > Préférences > Extensions, activer **OMFS Planner**.

### Segmentation IA (optionnelle, pour « Nouveau cas »)

Sans elle, on peut encore importer des STL déjà segmentés (Case > Autres imports).

1. Miniconda : <https://docs.anaconda.com/miniconda/>
2. L'environnement :
   ```sh
   conda create -n dentalseg python=3.12
   ~/miniconda3/envs/dentalseg/bin/pip install -r segmentation/requirements.txt
   ```
3. Les poids de DentalSegmentator (Dot et al., 2024 ; dossier `Dataset111_453CT`, depuis la page Zenodo du
   modèle), à copier dans `segmentation/model/`. Ils ne sont pas dans le dépôt (236 Mo, licence du modèle).
4. Si l'environnement ou le dossier sont ailleurs, indiquez-les dans les préférences de l'extension.

## Mises à jour

```sh
cd ~/Projects/omfs-planner
git pull
```

Relancez ensuite Blender. Vos cas ne sont jamais touchés : ils ne sont pas dans le dépôt.

## Données des patients

Aucune donnée de patient n'entre dans ce dépôt. Le `.gitignore` bloque les maillages, DICOM, volumes, `.blend`
et les dossiers `cases/` et `data/`. Les segmentations sont écrites à côté du CBCT, sur votre ordinateur.

## Développement

- Tests : `/Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup --python tests/run_tests.py`
- Architecture, règles et conventions : [CLAUDE.md](CLAUDE.md). Termes Blender ↔ Meshmixer : [docs/GLOSSARY.md](docs/GLOSSARY.md).

Licence : GPL-3.0-or-later (voir `blender_manifest.toml`).
