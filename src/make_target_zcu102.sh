echo "-----------------------------------------"
echo "MAKE TARGET ZCU102 STARTED.."
echo "-----------------------------------------"

TARGET_ZCU102=./target_zcu102
COMPILE_ZCU102=./compiled_model/zcu102 
APP=./application
NET_NAME=customcnn

# remove previous results
rm -rf ${TARGET_ZCU102}
mkdir -p ${TARGET_ZCU102}/model_dir

# copy application to TARGET_ZCU102 folder
cp ${APP}/*.py ${TARGET_ZCU102}
echo "  Copied application to TARGET_ZCU102 folder"


# copy xmodel to TARGET_ZCU102 folder
cp ${COMPILE_ZCU102}/${NET_NAME}.xmodel ${TARGET_ZCU102}/model_dir/.
echo "  Copied xmodel file(s) to TARGET_ZCU102 folder"

# create image files and copy to target folder
mkdir -p ${TARGET_ZCU102}/images

python generate_images.py  \
    --dataset=mnist \
    --image_dir=${TARGET_ZCU102}/images \
    --image_format=jpg \
    --max_images=10000

echo "  Copied images to TARGET_ZCU102 folder"

echo "-----------------------------------------"
echo "MAKE TARGET ZCU102 COMPLETED"
echo "-----------------------------------------"
