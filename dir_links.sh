# pass path to habitat data directory as argument
# e.g. bash dir_links.sh /home/ubuntu/habitat_data

ln -s $1 data

mkdir habitat-challenge-data
cd habitat-challenge-data
ln -s ../$1 data
