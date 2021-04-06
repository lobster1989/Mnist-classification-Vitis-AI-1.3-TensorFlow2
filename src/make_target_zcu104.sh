echo "-----------------------------------------"
echo "MAKE TARGET ZCU104 STARTED.."
echo "-----------------------------------------"

TARGET_ZCU104=./target_zcu104
COMPILE_ZCU104=./compiled_model/zcu104
APP=./application
NET_NAME=customcnn

# remove previous results
rm -rf ${TARGET_ZCU104}
mkdir -p ${TARGET_ZCU104}/model_dir

# copy application to TARGET_ZCU104 folder
cp ${APP}/*.py ${TARGET_ZCU104}
echo "  Copied application to TARGET_ZCU104 folder"


# copy xmodel to TARGET_ZCU104 folder
cp ${COMPILE_ZCU104}/${NET_NAME}.xmodel ${TARGET_ZCU104}/model_dir/.
echo "  Copied xmodel file(s) to TARGET_ZCU104 folder"

# create image files and copy to target folder
mkdir -p ${TARGET_ZCU104}/images

python generate_images.py  \
    --dataset=mnist \
    --image_dir=${TARGET_ZCU104}/images \
    --image_format=jpg \
    --max_images=10000

echo "  Copied images to TARGET_ZCU104 folder"

echo "-----------------------------------------"
echo "MAKE TARGET ZCU104 COMPLETED"
echo "-----------------------------------------"
