echo "1. Cloning code"
git clone https://github.com/lamhoangtung/MaskRCNN-card-crop-and-rotate
cd MaskRCNN-card-crop-and-rotate
echo "2. Installing dependencies"
sudo apt-get install -y build-essential libcap-dev
pip3 install -r requirements.txt
echo "3. Prepareing pretrained model"
mkdir data weights
cd weights
wget http://models.tensorpack.com/FasterRCNN/COCO-MaskRCNN-R50C41x.npz
echo "Done. Throw your data in and start training mate."
