#!/usr/bin/env bash
# Downloads the three public benchmarks from their ORIGINAL repositories into ./data (git-ignored).
# No data are redistributed by this repository. Pinned commits are the versions used in the study.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data && cd data

clone_at () {  # url dir commit
  if [ ! -d "$2" ]; then git clone "$1" "$2"; fi
  git -C "$2" checkout --quiet "$3"
}

# 1) FMC-MWO2KG (text, ISO 14224 failure modes) - Apache-2.0 repository
clone_at https://github.com/nlp-tlp/mwo2kg-and-echidna.git mwo2kg-and-echidna 96fc4477ea43a4bfd768430da386008b8803f1d1

# 2) Thermal images of an induction motor and a transformer (Najafi et al.) - images are inside .rar archives
clone_at https://github.com/mohnaj-nit/thermal-images-equip.git thermal-images-equip 8ac89a0b08f3f21f59a5922107164cca98d7e841
cd thermal-images-equip
mkdir -p ex_motor ex_trans
if   command -v bsdtar >/dev/null; then X="bsdtar -xf"; EXM="-C ex_motor"; EXT="-C ex_trans"
elif command -v unrar  >/dev/null; then X="unrar x -o+"; EXM=""; EXT=""
else echo "Install bsdtar (libarchive-tools) or unrar to extract the .rar archives"; exit 1; fi
if [ -z "$(ls -A ex_motor)" ]; then
  if [ "${X%% *}" = "bsdtar" ]; then bsdtar -xf IR-Motor-bmp.rar -C ex_motor; bsdtar -xf IR_trans_bmp.rar -C ex_trans
  else (cd ex_motor && unrar x -o+ ../IR-Motor-bmp.rar); (cd ex_trans && unrar x -o+ ../IR_trans_bmp.rar); fi
fi
cd ..

# 3) CWRU bearing data (GitHub mirror of the Case Western Reserve University Bearing Data Center; ~0.9 GB)
clone_at https://github.com/s-whynot/CWRU-dataset.git CWRU-dataset 0d07cb1caf0a5a68a955c2f43cfc21b9a208406a

echo "Done. Verify with:  python experiments/smoke_test.py"
