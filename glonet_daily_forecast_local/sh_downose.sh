export HOME=/Odyssey/private/$USER
source $HOME/.bashrc
conda activate glon

sleep 5
python down_copernicus.py 2025-07-11 -o /Odyssey/public/glonet/osse
sleep 5
python down_copernicus.py 2025-07-14 -o /Odyssey/public/glonet/osse
sleep 5
python down_copernicus.py 2025-07-17 -o /Odyssey/public/glonet/osse
sleep 5
python down_copernicus.py 2025-07-21 -o /Odyssey/public/glonet/osse

