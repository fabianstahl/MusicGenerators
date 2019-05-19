echo -n "Are you sure? (y/n)? "
read answer
if [ "$answer" != "${answer#[Yy]}" ] ;then
    sudo rm runs/* -rf
    sudo rm results/* -rf
    echo "Done cleaning runs/* and results/*"
else
    exit
fi
