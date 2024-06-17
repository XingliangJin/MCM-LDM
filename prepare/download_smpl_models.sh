mkdir -p deps/
cd deps/

echo "The smpl model will be stored in the './deps' folder"

# HumanAct12 poses
echo "Downloading"
gdown "https://drive.google.com/uc?id=1qrFkPZyRwRGd0Q3EY76K8oJaIgs_WK9i"
echo "Extracting"
tar xfzv smpl.tar.gz
echo "Cleaning"
rm smpl.tar.gz

# motionclip
mkdir -p motionclip/
cd motionclip/
gdown "https://drive.google.com/uc?id=1INYlGA76ak_cKGzvpOV2Pe6RkYTlXTW2"
unzip smpl.zip
echo -e "Cleaning\n"
rm smpl.zip



echo "Downloading done!"