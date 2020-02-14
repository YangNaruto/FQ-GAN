DATASET=$1

if [[$DATASET != "portrait" && $DATASET != "cat2dog"]]; then
  echo "dataset not available"
  exit
fi

URL=http://vllab.ucmerced.edu/hylee/DRIT/datasets/$DATASET.zip
wget -N $URL -O ../dataset/$DATASET.zip
unzip ../dataset/$DATASET.zip -d ../dataset
rm ../dataset/$DATASET.zip
